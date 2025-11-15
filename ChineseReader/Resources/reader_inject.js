 (function () {
   if (window.CR && window.CR.__ready) return;

   function addClass(cls) {
     if (!document.documentElement.classList.contains(cls)) {
       document.documentElement.classList.add(cls);
     }
   }

   function removeClass(cls) {
     document.documentElement.classList.remove(cls);
   }

   function injectStyleTag(cssText, id) {
     if (id && document.getElementById(id)) return;
     const style = document.createElement('style');
     if (id) style.id = id;
     style.type = 'text/css';
     style.appendChild(document.createTextNode(cssText));
     document.head.appendChild(style);
   }

   function isWhitespace(ch) {
     return /\s/.test(ch);
   }

   // Basic "is this a CJK Han character?"
   function isChinese(ch) {
     if (!ch) return false;
     const code = ch.charCodeAt(0);
     // CJK Unified Ideographs + Extension A + Compatibility Ideographs (good enough)
     return (
       (code >= 0x3400 && code <= 0x4DBF) || // Ext A
       (code >= 0x4E00 && code <= 0x9FFF) || // Basic
       (code >= 0xF900 && code <= 0xFAFF)    // Compatibility
     );
   }

   // "Normal" sentence boundaries
   function isSentenceBoundary(ch) {
     return /[。！？!?]/.test(ch);
   }

   // --- Jieba init management -------------------------------------------------
   function ensureJiebaInit() {
     if (!window.Jieba) return;

     // Already started or finished
     if (window.Jieba.__initStarted) return;

     window.Jieba.__initStarted = true;

     if (typeof window.Jieba.init === "function") {
       try {
         const p = window.Jieba.init();
         if (p && typeof p.then === "function") {
           window.Jieba.__initPromise = p
             .then(() => {
               window.Jieba.__ready = true;
               // console.log("[CR] Jieba initialized");
             })
             .catch((e) => {
               console.error("[CR] Jieba.init() failed", e);
             });
         } else {
           // init returned non-promise; assume ready immediately
           window.Jieba.__ready = true;
         }
       } catch (e) {
         console.error("[CR] Exception during Jieba.init()", e);
       }
     }
   }

   function jiebaCut(text) {
     // Kick off init in background when we first try to cut
     ensureJiebaInit();

     if (
       window.Jieba &&
       window.Jieba.__ready &&
       typeof window.Jieba.cut === "function"
     ) {
       try {
         return window.Jieba.cut(text) || [];
       } catch (e) {
         console.error("[CR] Jieba.cut failed, falling back", e);
         // Fall through to naive segmentation
       }
     }

     // Fallback: simple 1-char chunks
     const res = [];
     for (let i = 0; i < text.length; i += 1) {
       res.push(text.slice(i, i + 1));
     }
     return res;
   }

   // Idempotent segmentation: mark block with data-cr-segmented="1"
   function segmentBlockElement(block) {
     if (!block || block.dataset.crSegmented === "1") return;

     const walker = document.createTreeWalker(
       block,
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

     // Sentence-level state across the whole block
     /** Array< Array<string> > : per-sentence tokens (words + punctuation, no whitespace) */
     const allSentenceTokens = [];

     /** Tokens for the current sentence */
     let sentenceTokens = [];
     /** Word spans belonging to the current sentence (for assigning sentenceId) */
     let pendingWordSpans = [];
     /** Next sentence id within this block */
     let sentenceIdCounter = 0;
     /** Token index within current sentence (0-based, includes punctuation tokens) */
     let tokenIndexInSentence = 0;

     function commitSentence() {
       if (sentenceTokens.length === 0) {
         // Nothing meaningful in this sentence; just reset.
         pendingWordSpans = [];
         tokenIndexInSentence = 0;
         return;
       }

       const sentenceId = sentenceIdCounter;
       allSentenceTokens.push(sentenceTokens.slice());

       // Assign sentenceId to all word spans in this sentence
       for (const span of pendingWordSpans) {
         span.dataset.crSentenceId = String(sentenceId);
       }

       sentenceIdCounter += 1;
       sentenceTokens = [];
       pendingWordSpans = [];
       tokenIndexInSentence = 0;
     }

     textNodes.forEach((node) => {
       const text = node.nodeValue;
       const frag = document.createDocumentFragment();

       let i = 0;
       let chineseRun = "";

       function flushChineseRun() {
         if (!chineseRun) return;
         const words = jiebaCut(chineseRun);
         for (const w of words) {
           if (!w) continue;
           const span = document.createElement("span");
           span.textContent = w;
           span.className = "cr-word";
           span.dataset.crWord = "1";

           // Store token index (index into sentenceTokens)
           span.dataset.crTokenIndex = String(tokenIndexInSentence);

           // Append to DOM
           frag.appendChild(span);

           // Logical representation
           sentenceTokens.push(w);
           pendingWordSpans.push(span);
           tokenIndexInSentence += 1;
         }
         chineseRun = "";
       }

       while (i < text.length) {
         const ch = text[i];

         if (isWhitespace(ch)) {
           // Whitespace is kept visually but not added as a token
           flushChineseRun();
           frag.appendChild(document.createTextNode(ch));
           i += 1;
           continue;
         }

         if (isChinese(ch)) {
           chineseRun += ch;
           i += 1;
           continue;
         }

         // Non-Chinese, non-whitespace char: punctuation / Latin / etc.
         flushChineseRun();

         // Add the character to the DOM as-is
         frag.appendChild(document.createTextNode(ch));

         // Treat each such char as a token (e.g., punctuation, Latin letter)
         sentenceTokens.push(ch);
         tokenIndexInSentence += 1;

         // If this is a sentence boundary, finalize current sentence
         if (isSentenceBoundary(ch)) {
           commitSentence();
         }

         i += 1;
       }

       flushChineseRun(); // End of this text node

       if (node.parentNode) {
         node.parentNode.replaceChild(frag, node);
       }
     });

     // Final trailing sentence in the block, if any
     commitSentence();

     // Attach tokens to the block so we can retrieve them later
     block.__crSentenceTokens = allSentenceTokens;

     block.dataset.crSegmented = "1";
   }

   function findBlockElementFromPoint(x, y) {
     let el = document.elementFromPoint(x, y);
     if (!el) return null;

     if (el.nodeType === Node.TEXT_NODE) {
       el = el.parentElement;
     }
     if (!el) return null;

     return (
       el.closest("p, div, li, article, section, td, th") ||
       document.body
     );
   }

   function findWordSpanFromPoint(x, y) {
     let el = document.elementFromPoint(x, y);
     if (!el) return null;

     if (el.nodeType === Node.TEXT_NODE) {
       el = el.parentElement;
     }
     if (!el) return null;

     if (el.dataset && el.dataset.crWord === "1") {
       return el;
     }
     return el.closest('[data-cr-word="1"]');
   }

   function getWordInfoForSpan(span) {
     if (!span) return null;

     // Find the parent block that holds the sentence tokens
     const block = span.closest('[data-cr-segmented="1"]');
     if (!block) return null;

     const sentenceIdStr = span.dataset.crSentenceId;
     const tokenIndexStr = span.dataset.crTokenIndex;

     if (sentenceIdStr == null || tokenIndexStr == null) {
       return null;
     }

     const sentenceId = Number(sentenceIdStr);
     const wordIndex = Number(tokenIndexStr);

     const allSentenceTokens = block.__crSentenceTokens || [];
     const sentenceTokens = allSentenceTokens[sentenceId] || [];

     const rects = [];
     const domRects = span.getClientRects();
     for (let i = 0; i < domRects.length; i++) {
       const r = domRects[i];
       rects.push({
         x: r.left,
         y: r.top,
         width: r.width,
         height: r.height
       });
     }

     // New, compact format:
     // - sentenceTokens: array of tokens (words + punctuation, no whitespace)
     // - wordIndex: index into sentenceTokens
     // - rects: client rects
     return {
       sentenceTokens,
       wordIndex, // 0-based index in this sentence's token list
       rects
     };
   }

   window.CR = {
     __ready: true,
     _lastWordEl: null,

     setSelectable: function (selectable) {
       if (selectable) {
         removeClass("cr-nonselectable");
       } else {
         addClass("cr-nonselectable");
       }
     },

     // Main entry from Swift: highlight word under (x, y) and return info
     highlightWordAtPoint: function (x, y) {
       x = Number(x);
       y = Number(y);

       const block = findBlockElementFromPoint(x, y);
       if (!block) return null;

       // Ensure block is segmented; safe to call repeatedly
       segmentBlockElement(block);

       const span = findWordSpanFromPoint(x, y);
       if (!span) return null;

       const last = window.CR._lastWordEl;
       if (last && last !== span) {
         last.classList.remove("cr-word-highlight");
       }

       span.classList.add("cr-word-highlight");
       window.CR._lastWordEl = span;

       return getWordInfoForSpan(span);
     },
   };

   // Fallback CSS in case reader_inject.css wasn't added separately
   const fallbackCSS =
     "html.cr-nonselectable, html.cr-nonselectable body, html.cr-nonselectable *{" +
     "-webkit-user-select:none !important; user-select:none !important; -webkit-touch-callout:none !important; }" +
     ".cr-word-highlight{ background-color: rgba(120,170,255,0.4); border-radius:4px; }";

   injectStyleTag(fallbackCSS, "cr-nonselectable-fallback-style");

   // Optionally: kick off init *immediately* when this file loads,
   // even before the first segmentation.
   if (window.Jieba) {
     ensureJiebaInit();
   }
 })();
