// src/utils/omrPDFRenderer.ts
// ─── OMR PDF / Print renderer ────────────────────────────────────────────────
//
// v5.0 — layout matched exactly to AutoChecker bubble sheet photo reference:
//   • Header: "AutoChecker" centered, large bold; "BUBBLE ANSWER SHEET" centered
//     with wide letter-spacing below it.
//   • Field rows wrapped in a bordered box. Labels: Name / Section / Date / Score
//     on row 1; Subject / Test / Examination Teacher on row 2.
//   • Directions: bordered box with filled-circle "!" icon on the left.
//   • Info bar: AutoChecker © year  •  Examination Total Items: N  •  Exam ID: X
//     (sits between directions and bubble grid — replaces old footer).
//   • Bubble grid: plain white rows (no alternating shading), clean circles,
//     bold A B C D column headers, solid vertical divider between columns.
//   • Registration marks geometry preserved (scan-critical — do not change).

import * as FileSystem from 'expo-file-system';
import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';
import {
  getBubbleHeightPx,
  getColumnCount,
  getQNumWidthPx,
  OMR_CONFIG,
  OMROption,
  OMRSheetMeta,
  OUTER_PAD_PX
} from '../constants/omrConfig';

// ─── Paper size ───────────────────────────────────────────────────────────────
export type PaperSize = 'A4' | 'Legal';

const PAGE_CSS: Record<PaperSize, string> = {
  A4:    '210mm 297mm',
  Legal: '8.5in 13in',
};

const PAGE_WIDTH_PX: Record<PaperSize, number> = {
  A4:    794,
  Legal: 816,
};

// ─── Scan-critical constants (MUST match omrConfig.ts) ────────────────────────
const REG_SIZE = OMR_CONFIG.REG_SIZE_PX;  // 24
const REG_HALO = OMR_CONFIG.REG_HALO_PX; // 3
const outerPad = OUTER_PAD_PX;           // 33

// ─── Column helpers ───────────────────────────────────────────────────────────

function splitIntoColumns(total: number, cols: number): number[][] {
  const perCol = Math.ceil(total / cols);
  const result: number[][] = [];
  for (let c = 0; c < cols; c++) {
    const start = c * perCol + 1;
    const end   = Math.min(start + perCol - 1, total);
    if (start <= total) {
      result.push(Array.from({ length: end - start + 1 }, (_, i) => i + start));
    }
  }
  return result;
}

// ─── HTML generator ───────────────────────────────────────────────────────────

export function generateOMRHtml(meta: OMRSheetMeta, paperSize: PaperSize = 'A4'): string {
  const pageSize = PAGE_CSS[paperSize];
  const vpWidth  = PAGE_WIDTH_PX[paperSize];
  const vpHeight = paperSize === 'A4' ? 1123 : 1248;

  const totalQ  = meta.totalQuestions ?? OMR_CONFIG.TOTAL_QUESTIONS;
  const numCols = getColumnCount(totalQ);
  const columns = splitIntoColumns(totalQ, numCols);
  const perCol  = columns[0]?.length ?? Math.ceil(totalQ / numCols);

  // ── Geometry ──────────────────────────────────────────────────────────────
  const BODY_PAD     = 12;
  // Header section heights (approximate, empirically matched to photo):
  //   title block ≈ 48px, fields box ≈ 46px, directions box ≈ 38px, info bar ≈ 22px
  //   gaps/margins ≈ 16px  → total ≈ 170px
  const HEADER_H     = 170;
  const OUTER_PAD_TB = outerPad * 2;   // padding inside grid-wrap-outer top+bottom
  const COL_HDR_H    = 28;             // A B C D label row height inside each column
  const BOTTOM_SPACE = OMR_CONFIG.BOTTOM_GAP_PX;  // 50px breathing room below grid

  const availH = vpHeight - BODY_PAD * 2 - HEADER_H - OUTER_PAD_TB - COL_HDR_H - BOTTOM_SPACE;
  const rowH   = Math.floor(availH / perCol);

  const bubbleD = getBubbleHeightPx(totalQ);
  const qNumW   = getQNumWidthPx(totalQ);
  const qNumPt  = totalQ <= 25 ? 13 : totalQ <= 50 ? 12 : totalQ <= 75 ? 11 : 9;

  const year = new Date().getFullYear();

  function colHeader(): string {
    return `
      <div class="col-header">
        <span style="width:${qNumW + 4}px;flex-shrink:0;display:inline-block"></span>
        ${OMR_CONFIG.OPTIONS.map((o: OMROption) =>
          `<span class="opt-hdr">${o}</span>`
        ).join('')}
      </div>`;
  }

  function renderColumn(questions: number[]): string {
    return questions.map((qNum) => `
      <div class="q-row" style="height:${rowH}px">
        <span class="q-num" style="width:${qNumW}px">${qNum}.</span>
        <div class="bubbles">
          ${OMR_CONFIG.OPTIONS.map((opt: OMROption) =>
            `<span class="bubble"
              style="width:${bubbleD}px;height:${bubbleD}px"
              data-q="${qNum}" data-opt="${opt}"></span>`
          ).join('')}
        </div>
      </div>`
    ).join('');
  }

  return `<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=${vpWidth}"/>
<style>
  @page { size: ${pageSize}; margin: 0; }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  html { width: ${vpWidth}px; height: ${vpHeight}px; }
  body {
    width: ${vpWidth}px;
    height: ${vpHeight}px;
    background: #fff;
    font-family: Arial, Helvetica, sans-serif;
    color: #000;
    padding: ${BODY_PAD}px;
    position: relative;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ── Outer page border ─────────────────────────────────────────────────── */
  .page-border {
    position: absolute; inset: 6px;
    border: 1.5px solid #000;
    pointer-events: none;
    z-index: 0;
  }

  /* ── Title block — centered ────────────────────────────────────────────── */
  .title-block {
    text-align: center;
    margin-top: 4px;
    margin-bottom: 5px;
    flex-shrink: 0;
  }
  .app-name {
    font-size: 22pt;
    font-weight: 900;
    letter-spacing: 0.5px;
    line-height: 1.1;
  }
  .sheet-label {
    font-size: 7pt;
    font-weight: 700;
    letter-spacing: 5px;
    color: #000;
    margin-top: 2px;
  }

  /* ── Fields — wrapped in thin border box ───────────────────────────────── */
  .fields-wrap {
    border: 1px solid #aaa;
    padding: 5px 8px;
    margin-bottom: 5px;
    flex-shrink: 0;
  }
  .field-row {
    display: flex;
    align-items: baseline;
    gap: 6px;
    margin-bottom: 5px;
  }
  .field-row:last-child { margin-bottom: 0; }
  .fl  { font-size: 7.5pt; font-weight: 700; white-space: nowrap; flex-shrink: 0; }
  .fln {
    border-bottom: 1px solid #000;
    flex: 1;
    min-height: 15px;
    font-size: 7.5pt;
    padding-bottom: 1px;
  }

  /* ── Directions box ────────────────────────────────────────────────────── */
  .directions-box {
    border: 1.5px solid #000;
    border-radius: 2px;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 10px;
    margin-bottom: 4px;
    flex-shrink: 0;
  }
  .dir-icon {
    width: 22px;
    height: 22px;
    border-radius: 50%;
    background: #000;
    color: #fff;
    font-size: 12pt;
    font-weight: 900;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    line-height: 1;
    padding-bottom: 1px;
  }
  .dir-text {
    font-size: 7pt;
    font-weight: 400;
    color: #000;
    text-align: center;
    flex: 1;
    line-height: 1.6;
  }

  /* ── Info bar ──────────────────────────────────────────────────────────── */
  .info-bar {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    font-size: 6.5pt;
    color: #000;
    padding: 3px 4px;
    border-top: 0.8px solid #bbb;
    border-bottom: 0.8px solid #bbb;
    margin-bottom: 0;
    flex-shrink: 0;
  }
  .info-dot { color: #000; font-size: 8pt; }

  /* ── Registration marks + grid wrapper ─────────────────────────────────── */
  .grid-wrap-outer {
    position: relative;
    padding: ${outerPad}px;
    flex: 1;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .fid {
    position: absolute;
    width:  ${REG_SIZE}px;
    height: ${REG_SIZE}px;
    background: #000;
    border-radius: 0;
    box-shadow: 0 0 0 ${REG_HALO}px #fff, 0 0 0 ${REG_HALO + 0.5}px #ddd;
    z-index: 2;
  }
  .fid-tl { top:    ${REG_HALO}px; left:   ${REG_HALO}px; }
  .fid-tr { top:    ${REG_HALO}px; right:  ${REG_HALO}px; }
  .fid-bl { bottom: ${REG_HALO}px; left:   ${REG_HALO}px; }
  .fid-br { bottom: ${REG_HALO}px; right:  ${REG_HALO}px; }

  /* ── Bubble grid ───────────────────────────────────────────────────────── */
  .grid-wrap {
    display: flex;
    border: 1px solid #999;
    width: 100%;
    flex: 1;
    min-height: 0;
  }
  .grid-col {
    flex: 1;
    padding: 2px 6px;
    min-width: 0;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .col-sep {
    width: 1px;
    flex-shrink: 0;
    background: #888;
  }

  /* Column header: A B C D */
  .col-header {
    display: flex;
    align-items: center;
    flex-shrink: 0;
    border-bottom: 1px solid #aaa;
    padding-bottom: 4px;
    margin-bottom: 2px;
  }
  .opt-hdr {
    flex: 1;
    text-align: center;
    font-weight: 900;
    color: #000;
    font-size: ${qNumPt}pt;
  }

  /* Question rows — plain white, no alternating shading */
  .q-rows { display: flex; flex-direction: column; }
  .q-row {
    display: flex;
    align-items: center;
    overflow: hidden;
    flex-shrink: 0;
    background: #fff;
  }
  .q-num {
    font-weight: 700;
    flex-shrink: 0;
    color: #000;
    line-height: 1;
    font-size: ${qNumPt}pt;
    text-align: right;
    padding-right: 4px;
  }

  /* Circles evenly spaced */
  .bubbles {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: space-around;
    padding: 0 2px;
  }
  .bubble {
    border-radius: 50%;
    border: 1.5px solid #333;
    background: #fff;
    flex-shrink: 0 !important;
    flex-grow: 0 !important;
    display: inline-block;
  }

  /* Bottom breathing room */
  .bottom-gap { height: ${BOTTOM_SPACE}px; flex-shrink: 0; }
</style>
</head>
<body>

  <div class="page-border"></div>

  <!-- ── Title ──────────────────────────────────────────────────────────── -->
  <div class="title-block">
    <div class="app-name">AutoChecker</div>
    <div class="sheet-label">B U B B L E &nbsp;&nbsp; A N S W E R &nbsp;&nbsp; S H E E T</div>
  </div>

  <!-- ── Fields ─────────────────────────────────────────────────────────── -->
  <div class="fields-wrap">
    <div class="field-row">
      <span class="fl">Name:</span>
      <span class="fln" style="flex:2">${meta.studentName ?? ''}</span>
      <span class="fl">Section:</span>
      <span class="fln" style="max-width:100px">${meta.section ?? ''}</span>
      <span class="fl">Date:</span>
      <span class="fln" style="max-width:88px">${meta.date ?? ''}</span>
      <span class="fl">Score:</span>
      <span class="fln" style="max-width:55px"></span>
    </div>
    <div class="field-row">
      <span class="fl">Subject:</span>
      <span class="fln" style="flex:2;max-width:180px">${meta.subject ?? ''}</span>
      <span class="fl">Test:</span>
      <span class="fln" style="flex:2;max-width:150px">${meta.testTitle ?? ''}</span>
      <span class="fl">Examination Teacher:</span>
      <span class="fln"></span>
    </div>
  </div>

  <!-- ── Directions ─────────────────────────────────────────────────────── -->
  <div class="directions-box">
    <div class="dir-icon">!</div>
    <div class="dir-text">
      <b>DIRECTIONS:</b> Use a <b>BLACK</b> ballpen. Shade <b>ONLY ONE</b> bubble per item completely.
      Do <b>NOT</b> bend or fold this sheet.
    </div>
  </div>

  <!-- ── Info bar ───────────────────────────────────────────────────────── -->
  <div class="info-bar">
    <span>AutoChecker &copy; ${year}</span>
    <span class="info-dot">&bull;</span>
    <span>Examination Total Items: ${totalQ}</span>
    <span class="info-dot">&bull;</span>
    <span>Exam ID: ${meta.examId ?? '&mdash;'}</span>
  </div>

  <!-- ── Bubble grid ─────────────────────────────────────────────────────── -->
  <div class="grid-wrap-outer">

    <div class="fid fid-tl"></div>
    <div class="fid fid-tr"></div>
    <div class="fid fid-bl"></div>
    <div class="fid fid-br"></div>

    <div class="grid-wrap">
      ${columns.map((qs, ci) => `
        ${ci > 0 ? '<div class="col-sep"></div>' : ''}
        <div class="grid-col">
          ${colHeader()}
          <div class="q-rows">
            ${renderColumn(qs)}
          </div>
        </div>
      `).join('')}
    </div>

  </div>

  <div class="bottom-gap"></div>

</body>
</html>`;
}

// ─── Print & Share ────────────────────────────────────────────────────────────

export async function printOMRSheet(
  meta: OMRSheetMeta,
  paperSize: PaperSize = OMR_CONFIG.DEFAULT_PAPER_SIZE,
): Promise<void> {
  const html = generateOMRHtml(meta, paperSize);
  await Print.printAsync({ html });
}

export async function shareOMRSheet(
  meta: OMRSheetMeta,
  paperSize: PaperSize = OMR_CONFIG.DEFAULT_PAPER_SIZE,
): Promise<void> {
  const html = generateOMRHtml(meta, paperSize);

  const safeName = `AutoChecker-OMR-Bubble-${(meta.examId ?? 'Sheet').replace(/[^a-zA-Z0-9\-_]/g, '')}`;
  const { uri } = await Print.printToFileAsync({ html, base64: false });

  const destUri = `${FileSystem.cacheDirectory}${safeName}.pdf`;
  await FileSystem.moveAsync({ from: uri, to: destUri });

  const canShare = await Sharing.isAvailableAsync();
  if (canShare) {
    await Sharing.shareAsync(destUri, {
      mimeType:    'application/pdf',
      dialogTitle: `AutoChecker OMR Bubble Sheet — ${meta.testTitle}`,
      UTI:         'com.adobe.pdf',
    });
  }
}