// vinAgent.js
// Run: node vinAgent.js --q="WVWZZZ1JZXW000001"

import 'dotenv/config'
import axios from 'axios'
import { ChatOpenAI } from '@langchain/openai'
import { StateGraph, START, END, Annotation } from '@langchain/langgraph'
import { tavily } from '@tavily/core'

// ──────────────────────────────────────────────────────────────────────────────
// ТУТОРІАЛ: ЩО ТУТ ВІДБУВАЄТЬСЯ
// 1) LangGraph = граф виконання з вузлів (node) і ребер (edge). Кожен вузол — це
//    звичайна async-функція, що приймає "стан" (state) і повертає патч стану.
// 2) "Стан" (State) — це єдина «правда» агента: вхідні дані, проміжні агрегації,
//    результати API, ознаки ризиків тощо. Annotation() описує «схему» полів.
// 3) Ми визначаємо вузли-процедури: normalize → vin_info → web_search → risks →
//    markers → report. Кожен вузол додає/оновлює щось у state.
// 4) Далі «малюємо» ребра (послідовність), компілюємо граф і викликаємо його.
// 5) У цьому прикладі зовнішні інтеграції: NHTSA (VIN decode) і Tavily (веб-пошук).
// ──────────────────────────────────────────────────────────────────────────────

// ── Helpers ─────────────────────────────────────────────
const NON_ALNUM = /[^A-Za-z0-9]/g
function normalizeQuery(q) {
  const v = (q || '').trim().toUpperCase().replace(NON_ALNUM, '')
  if (isLikelyVIN(v)) return { kind: 'vin', value: v }
  if (/^[A-ZА-ЯІЇЄҐ0-9]{5,10}$/.test(v)) return { kind: 'plate', value: v }
  return { kind: 'unknown', value: v }
}
function isLikelyVIN(v) { return /^[A-HJ-NPR-Z0-9]{17}$/.test(v) && vinChecksumValid(v) }
// VIN checksum ISO 3779
const translit = {A:1,B:2,C:3,D:4,E:5,F:6,G:7,H:8,J:1,K:2,L:3,M:4,N:5,P:7,R:9,S:2,T:3,U:4,V:5,W:6,X:7,Y:8,Z:9,'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
const weights = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]
function vinChecksumValid(vin) {
  if (!/^[A-HJ-NPR-Z0-9]{17}$/.test(vin)) return false
  const sum = vin.split('').reduce((acc, ch, i) => acc + (translit[ch] ?? 0) * weights[i], 0)
  const remainder = sum % 11
  const check = remainder === 10 ? 'X' : String(remainder)
  return vin[8] === check
}

const RISK_KEYWORDS = [
  'salvage','totaled','accident','crash','flood','hail','fire','theft','stolen',
  'copart','iaai','auction','odometer rollback','mileage rollback','write-off','damaged','repairable'
]

// ── External calls ──────────────────────────────────────
// 1) VIN decode через офіційний NHTSA API
async function nhtsaDecodeVin(vin) {
  const url = `https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/${vin}?format=json`
  const { data } = await axios.get(url, { timeout: 20000 })
  return data
}

// 2) Веб-пошук через TAVILY (заміна SerpAPI)
//    • tavily SDK повертає об’єкт із масивом results: [{ title, url, content, ... }, ...]
//    • Для сумісності решти коду нормалізуємо до { title, link, snippet }.
const tv = tavily({ apiKey: process.env.TAVILY_API_KEY })
async function webSearch(query) {
  if (!process.env.TAVILY_API_KEY) throw new Error('Set TAVILY_API_KEY')
  // Можна передавати просто рядок або об’єкт із опціями.
  // Тут показуємо «просунутий» варіант із фільтрами:
  const res = await tv.search(query)
  const items = (res.results || []).map(r => ({
    title: r.title,
    link:  r.url,
    snippet: r.content
  }))
  return items
}

// ── State ───────────────────────────────────────────────
// Annotation.Root описує «форму» загального стану графа.
// • input: сирий ввід (рядок із CLI)
// • norm: нормалізований ввід (vin|plate|unknown + значення)
// • aggregates: агреговані артефакти між вузлами
const State = Annotation.Root({
  input: Annotation(),         // { q }
  norm: Annotation(),          // { kind, value }
  aggregates: Annotation({ default: {} }) // { vin, plate, vinValid, nhtsa, webHits, risks, markers, report, facts }
})

// ── LLM ─────────────────────────────────────────────────
// LangChain обгортає клієнт OpenAI, який ми використовуємо в кінці для короткого звіту.
const model = new ChatOpenAI({ model: 'gpt-5-nano' })

// ── Nodes (вузли графа) ─────────────────────────────────
// Вузол = pure async-функція(state) → повертає *патч* стану (лише те, що змінює).

/** Нормалізація вводу: визначити, це VIN чи номерний знак, або «unknown». */
async function nodeNormalize(state) {
  const { q } = state.input
  const norm = normalizeQuery(q)
  const aggregates = { ...(state.aggregates || {}) }
  if (norm.kind === 'vin') { aggregates.vin = norm.value; aggregates.vinValid = vinChecksumValid(norm.value) }
  if (norm.kind === 'plate') { aggregates.plate = norm.value }
  return { norm, aggregates }
}

/** VIN → NHTSA decode + початкові факти для звіту. */
async function nodeVinInfo(state) {
  const { norm, aggregates } = state
  if (norm.kind !== 'vin' || !aggregates.vin) return {}
  const nhtsa = await nhtsaDecodeVin(aggregates.vin)
  const facts = { ...(aggregates.facts || {}) }
  try {
    const row = nhtsa?.Results?.[0] || {}
    facts.Make = row.Make
    facts.Model = row.Model
    facts.ModelYear = row.ModelYear
    facts.BodyClass = row.BodyClass
    facts.VehicleType = row.VehicleType
    facts.PlantCountry = row.PlantCountry
  } catch {}
  return { aggregates: { ...aggregates, nhtsa, facts } }
}

/** Веб-пошук через Tavily. Для VIN шукаємо за самим VIN; інакше — plate або сирий ввід. */
async function nodeWebSearch(state) {
  const { norm, aggregates, input } = state
  const query = norm.kind === 'vin' ? aggregates.vin : (aggregates.plate || input.q)
  const webHits = await webSearch(query)
  return { aggregates: { ...aggregates, webHits } }
}

/** Дешеві «rule-based» ризики по ключових словах і патернах у webHits/NHTSA. */
async function nodeRisks(state) {
  const { aggregates } = state
  const text = JSON.stringify(aggregates.webHits || []) + ' ' + JSON.stringify(aggregates.nhtsa || {})
  const risks = new Set()
  for (const kw of RISK_KEYWORDS) if (text.toLowerCase().includes(kw)) risks.add(kw)
  const auctionCount = (aggregates.webHits || []).filter(h => /copart|iaai|auction|salvage/i.test(h.link || '')).length
  if (auctionCount >= 2) risks.add('multiple_auctions')
  return { aggregates: { ...aggregates, risks: Array.from(risks) } }
}

/** Маркери як «чек-лист» для операторів: що добре/погано, короткі нотатки. */
async function nodeMarkers(state) {
  const { norm, aggregates } = state
  const markers = {}
  markers.input_type   = { ok: norm.kind !== 'unknown', note: norm.kind }
  if (norm.kind === 'vin') {
    markers.vin_checksum = { ok: !!aggregates.vinValid, note: aggregates.vin }
    const decodedOk = !!(aggregates.nhtsa?.Results?.[0]?.Model)
    markers.vin_decoded  = { ok: decodedOk, note: decodedOk ? 'NHTSA decoded' : 'no decode' }
  }
  const hasWeb = (aggregates.webHits || []).length > 0
  markers.web_presence = { ok: hasWeb, note: hasWeb ? `${aggregates.webHits.length} hits` : 'no hits' }
  const risky = (aggregates.risks || []).length > 0
  markers.risk_flags   = { ok: !risky, note: risky ? (aggregates.risks || []).join(', ') : 'none' }
  return { aggregates: { ...aggregates, markers } }
}

/** Короткий звіт через LLM (LangChain + OpenAI): маркери, факти, джерела. */
async function nodeReport(state) {
  const { aggregates } = state
  const res = await model.invoke([
    { role: 'system', content: 'Summarize vehicle intelligence for AUTO.ria operators. Be precise, fact-first.' },
    { role: 'user', content:
      `Create a short report: 
       1) Markers (bullet list with OK/WARN), 
       2) Key facts (make/model/year/body/etc), 
       3) 3–8 sources (title + domain).
       Data: ${JSON.stringify({
         vin: aggregates.vin,
         plate: aggregates.plate,
         facts: aggregates.facts,
         markers: aggregates.markers,
         hits: (aggregates.webHits || []).slice(0,8)
       })}`
    }
  ])
  return { aggregates: { ...aggregates, report: res.content } }
}

// ── Graph (побудова конвеєра) ───────────────────────────
// 1) Оголошуємо граф із «схемою» стану.
// 2) Реєструємо вузли.
// 3) Малюємо ребра (послідовність виконання).
// 4) compile() повертає готовий застосунок (executor).
const graph = new StateGraph(State)
  .addNode('normalize', nodeNormalize)
  .addNode('vin_info', nodeVinInfo)
  .addNode('web_search', nodeWebSearch)
  .addNode('risks', nodeRisks)
  .addNode('markers', nodeMarkers)
  .addNode('report', nodeReport)
  .addEdge(START, 'normalize')
  .addEdge('normalize', 'vin_info')
  .addEdge('vin_info', 'web_search')
  .addEdge('web_search', 'risks')
  .addEdge('risks', 'markers')
  .addEdge('markers', 'report')
  .addEdge('report', END)

const app = graph.compile()

// ── CLI ────────────────────────────────────────────────
// Запуск: node vinAgent.js --q="<VIN or plate>"
const arg = process.argv.find(a => a.startsWith('--q='))
if (!arg) { console.error('Usage: node vinAgent.js --q="<VIN or plate>"'); process.exit(1) }
const q = arg.slice(4)

// Виконання графа: на вхід подаємо state.input, на виході отримаємо повний state.
app.invoke({ input: { q } })
  .then(out => {
    const ag = out.aggregates || {}
    console.log('\n=== MARKERS ===')
    for (const [k,v] of Object.entries(ag.markers || {})) {
      console.log(`- ${k}: ${v.ok ? 'OK' : 'WARN'}${v.note ? ` (${v.note})` : ''}`)
    }
    console.log('\n=== FACTS ===')
    console.log(JSON.stringify(ag.facts || {}, null, 2))
    console.log('\n=== SOURCES ===')
    for (const h of (ag.webHits || []).slice(0,8)) {
      try {
        const domain = new URL(h.link).hostname
        console.log(`- ${h.title} [${domain}]`)
      } catch { console.log(`- ${h.title} [link]`) }
    }
    console.log('\n=== REPORT ===')
    console.log(ag.report || '(no report)')
  })
  .catch(e => { console.error('Fatal:', e?.response?.data || e); process.exit(1) })
