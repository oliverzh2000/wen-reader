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

     textNodes.forEach((node) => {
       const text = node.nodeValue;
       const frag = document.createDocumentFragment();
       let i = 0;

       while (i < text.length) {
         const ch = text[i];
         if (isWhitespace(ch)) {
           frag.appendChild(document.createTextNode(ch));
           i += 1;
           continue;
         }

         const word = text.slice(i, i + 2); // simple 2-char "word" for now
         const span = document.createElement("span");
         span.textContent = word;
         span.className = "cr-word";
         span.dataset.crWord = "1";
         frag.appendChild(span);
         i += 2;
       }

       if (node.parentNode) {
         node.parentNode.replaceChild(frag, node);
       }
     });

     block.dataset.crSegmented = "1";
   }

   function findBlockElementFromPoint(x, y) {
     let el = document.elementFromPoint(x, y);
     if (!el) return null;

     // elementFromPoint never returns text nodes in modern browsers,
     // but just in case:
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

     // Main entry point from Swift: highlight word under (x, y) in viewport coords
     highlightWordAtPoint: function (x, y) {
       x = Number(x);
       y = Number(y);

       const block = findBlockElementFromPoint(x, y);
       if (!block) return;

       // Ensure block is segmented; safe to call repeatedly
       segmentBlockElement(block);

       const span = findWordSpanFromPoint(x, y);
       if (!span) return;

       const last = window.CR._lastWordEl;
       if (last && last !== span) {
         last.classList.remove("cr-word-highlight");
       }

       span.classList.add("cr-word-highlight");
       window.CR._lastWordEl = span;
     },
   };

   // Fallback CSS in case reader_inject.css wasn't added separately
   const fallbackCSS =
     "html.cr-nonselectable, html.cr-nonselectable body, html.cr-nonselectable *{" +
     "-webkit-user-select:none !important; user-select:none !important; -webkit-touch-callout:none !important; }" +
     ".cr-word-highlight{ background-color: rgba(120,170,255,0.4); border-radius:4px; }";

   injectStyleTag(fallbackCSS, "cr-nonselectable-fallback-style");
 })();
