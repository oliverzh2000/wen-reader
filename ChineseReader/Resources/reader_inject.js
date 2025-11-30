 (function () {
   if (window.CR && window.CR.__ready) return;

   // ---------- Utilities ----------

   function addClass(cls) {
     if (!document.documentElement.classList.contains(cls)) {
       document.documentElement.classList.add(cls);
     }
   }

   function removeClass(cls) {
     document.documentElement.classList.remove(cls);
   }

   function isWhitespace(ch) {
     return /\s/.test(ch);
   }

   function isChinese(ch) {
     if (!ch) return false;
     const code = ch.charCodeAt(0);
     return (
       (code >= 0x3400 && code <= 0x4DBF) || // Ext A
       (code >= 0x4E00 && code <= 0x9FFF) || // Basic
       (code >= 0xF900 && code <= 0xFAFF)    // Compatibility
     );
   }

   function isSentenceBoundary(ch) {
     return /[。！？!?]/.test(ch);
   }

   function elementFromPointSafe(x, y) {
     let el = document.elementFromPoint(x, y);
     if (!el) return null;
     if (el.nodeType === Node.TEXT_NODE) {
       el = el.parentElement;
     }
     return el;
   }

   function findBlockElementFromPoint(x, y) {
     const el = elementFromPointSafe(x, y);
     if (!el) return null;
     return (
       el.closest("p, div, li, article, section, td, th") ||
       document.body
     );
   }

   // ---------- Generic segmentation helpers ----------

   /**
    * Generic helper: walk text nodes under `root`, call `segmentFn(node, text)`
    * which returns a DocumentFragment to replace that text node with.
    * Idempotency is controlled via `flagAttr` on root.
    */
   function segmentTextNodesOnce(root, flagAttr, segmentFn) {
     if (!root || root.dataset[flagAttr] === "1") return;

     const walker = document.createTreeWalker(
       root,
       NodeFilter.SHOW_TEXT,
       {
         acceptNode(node) {
           if (!node.nodeValue || !node.nodeValue.trim()) {
             return NodeFilter.FILTER_REJECT;
           }
           return NodeFilter.FILTER_ACCEPT;
         }
       }
     );

     const textNodes = [];
     while (walker.nextNode()) {
       textNodes.push(walker.currentNode);
     }

     for (const node of textNodes) {
       const text = node.nodeValue;
       if (!text) continue;
       const frag = segmentFn(node, text);
       if (frag && node.parentNode) {
         node.parentNode.replaceChild(frag, node);
       }
     }

     root.dataset[flagAttr] = "1";
   }

   // ---------- Sentence segmentation ----------
     function ensureSentenceSpans(block) {
       segmentTextNodesOnce(block, "crSentWrapped", function (_node, text) {
         const frag = document.createDocumentFragment();
         let currentSentenceSpan = null;
         let buffer = "";

         function ensureSentenceSpan() {
           if (!currentSentenceSpan) {
             currentSentenceSpan = document.createElement("span");
             currentSentenceSpan.className = "cr-sentence";
             frag.appendChild(currentSentenceSpan);
           }
           return currentSentenceSpan;
         }

         function flushBuffer() {
           if (!buffer) return;
           const span = ensureSentenceSpan();
           span.appendChild(document.createTextNode(buffer));
           buffer = "";
         }

         for (let i = 0; i < text.length; i++) {
           const ch = text[i];

           if (isWhitespace(ch)) {
             // End of any current sentence chunk; whitespace lives outside
             flushBuffer();
             currentSentenceSpan = null;
             frag.appendChild(document.createTextNode(ch));
             continue;
           }

           if (isSentenceBoundary(ch)) {
             // Include the boundary char in this sentence
             buffer += ch;
             flushBuffer();
             currentSentenceSpan = null; // next non-whitespace starts a new sentence
           } else {
             // Normal char: just accumulate
             buffer += ch;
           }
         }

         // Flush trailing text if any
         flushBuffer();

         return frag;
       });
     }


   function sentenceFromPoint(x, y, block) {
     const el = elementFromPointSafe(x, y);
     if (!el) return null;
     return el.closest(".cr-sentence") || block;
   }

   // ---------- Chinese run segmentation ----------

   function ensureRunSpans(sentenceEl) {
     segmentTextNodesOnce(sentenceEl, "crRunWrapped", function (_node, text) {
       const frag = document.createDocumentFragment();
       let run = "";

       function flushRun() {
         if (!run) return;
         const span = document.createElement("span");
         span.className = "cr-run";
         span.dataset.crRun = "1";
         span.dataset.crOriginalText = run;
         span.textContent = run;
         frag.appendChild(span);
         run = "";
       }

       for (let i = 0; i < text.length; i++) {
         const ch = text[i];

         if (isChinese(ch)) {
           run += ch;
         } else {
           flushRun();
           frag.appendChild(document.createTextNode(ch));
         }
       }

       flushRun();
       return frag;
     });
   }

   function runFromPoint(x, y) {
     const el = elementFromPointSafe(x, y);
     if (!el) return null;
     if (el.dataset && el.dataset.crRun === "1") {
       return el;
     }
     return el.closest('[data-cr-run="1"]');
   }

   function markLastRun(runSpan) {
     if (!runSpan) return null;

     if (window.CR._lastRunEl && window.CR._lastRunEl !== runSpan) {
       if (window.CR._lastRunEl.id === "cr-current-run") {
         window.CR._lastRunEl.id = "";
       }
     }

     window.CR._lastRunEl = runSpan;
     runSpan.id = "cr-current-run";
     return runSpan.id;
   }

   // ---------- Word segmentation (using lengths) ----------

   function ensureWordsForRun(runSpan, lengths) {
     if (!runSpan) return;

     const text =
       runSpan.dataset.crOriginalText != null
         ? runSpan.dataset.crOriginalText
         : runSpan.textContent || "";

     // Idempotent: always rebuild from original text & lengths
     while (runSpan.firstChild) {
       runSpan.removeChild(runSpan.firstChild);
     }

     let offset = 0;
     lengths = Array.isArray(lengths) ? lengths : [];

     for (let i = 0; i < lengths.length; i++) {
       const len = lengths[i] | 0;
       if (len <= 0) continue;
       const end = offset + len;
       if (end > text.length) break;

       const part = text.slice(offset, end);
       const span = document.createElement("span");
       span.className = "cr-word";
       span.dataset.crWord = "1";
       span.textContent = part;
       runSpan.appendChild(span);

       offset = end;
     }

     // Any leftover chars (e.g., lengths don't cover the whole run) stay as plain text.
     if (offset < text.length) {
       runSpan.appendChild(
         document.createTextNode(text.slice(offset))
       );
     }

     runSpan.dataset.crWordsWrapped = "1";
   }

   function wordSpanFromPoint(x, y) {
     const el = elementFromPointSafe(x, y);
     if (!el) return null;
     if (el.dataset && el.dataset.crWord === "1") return el;
     return el.closest('[data-cr-word="1"]');
   }

   // ---------- Public API ----------

   window.CR = {
     __ready: true,
     _lastRunEl: null,
     _lastWordEl: null,

     // keep selectable toggling
     setSelectable: function (selectable) {
       if (selectable) {
         removeClass("cr-nonselectable");
       } else {
         addClass("cr-nonselectable");
       }
     },

     clearHighlight: function () {
       const last = window.CR._lastWordEl;
       if (last) {
         last.classList.remove("cr-word-highlight");
         window.CR._lastWordEl = null;
       }
     },

     /**
      * 1) Return context around (x, y).
      * - Finds block, sentence, Chinese run.
      * - Gives the run a stable ID for later (segmentAndHighlight).
      * - Returns plain strings.
      *
      * Returns:
      *   { block: string, sentence: string, run: string, runId: string|null }
      * or null on failure.
      */
     getContextAtPoint: function (x, y) {
       x = Number(x);
       y = Number(y);

       const block = findBlockElementFromPoint(x, y);
       if (!block) return null;

       // 1) Ensure sentence spans live inside the block
       ensureSentenceSpans(block);
       const sentence = sentenceFromPoint(x, y, block) || block;

       // 2) Ensure Chinese runs live inside that sentence
       ensureRunSpans(sentence);
       const run = runFromPoint(x, y);

       const runId = run ? markLastRun(run) : null;

       return {
         block: block.innerText || "",
         sentence: sentence.innerText || "",
         run: run ? (run.textContent || "") : "",
         runId: runId
       };
     },

     /**
      * 2) Segment the last run into words (using Swift-provided lengths)
      *    and highlight the word under (x, y).
      *
      * Swift drives segmentation; JS only:
      *   - Rebuilds spans for that run
      *   - Chooses the word under the finger
      *   - Applies .cr-word-highlight
      */
     segmentAndHighlightAtPoint: function (x, y, lengths) {
       x = Number(x);
       y = Number(y);

       const run =
         window.CR._lastRunEl ||
         document.getElementById("cr-current-run");

       if (!run) {
         // No known run yet; nothing to do.
         return;
       }

       // Idempotent segmentation into .cr-word spans
       ensureWordsForRun(run, lengths);

       // Clear old highlight (also idempotent)
       window.CR.clearHighlight();

       const wordSpan = wordSpanFromPoint(x, y);
       if (!wordSpan) return;

       wordSpan.classList.add("cr-word-highlight");
       window.CR._lastWordEl = wordSpan;
         
         // Prepare rects as return value.
         const rects = [];
         const domRects = wordSpan.getClientRects();
         for (let i = 0; i < domRects.length; i++) {
           const r = domRects[i];
           rects.push({
             x: r.left,
             y: r.top,
             width: r.width,
             height: r.height
           });
         }
         
         return {
             word: wordSpan.textContent,
             rects: rects
         }
     }
   };
 })();
