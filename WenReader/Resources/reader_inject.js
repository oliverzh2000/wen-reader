(function () {
  if (window.WR && window.WR.__ready) return;

  // ---------- Constants ----------

  var HIGHLIGHT_ID = "wr-highlight";
  var ACTIVE_BLOCK_ID = "wr-active-block";
  var BLOCK_SELECTOR = "p, div, li, blockquote, td, th, article, section";

  // ---------- Utilities ----------

  function findBlockFromPoint(x, y) {
    var el = document.elementFromPoint(x, y);
    if (!el) return null;
    if (el.nodeType === Node.TEXT_NODE) el = el.parentElement;
    // Walk up to find the nearest block-level text container
    var block = el.closest(BLOCK_SELECTOR);
    return block || null;
  }

  /**
   * Get the full text content of a block, joining all text nodes in order.
   * Returns { text, textNodes } where textNodes is array of { node, startOffset }
   * mapping each text node to its character offset in the combined string.
   */
  function getBlockTextMap(block) {
    var walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT, null);
    var textNodes = [];
    var fullText = "";
    var node;
    while ((node = walker.nextNode())) {
      // Skip the highlight span's content — we'll count its text but note it's inside highlight
      textNodes.push({ node: node, startOffset: fullText.length });
      fullText += node.nodeValue || "";
    }
    return { text: fullText, textNodes: textNodes };
  }

  /**
   * Temporarily wrap every character in the block into individual spans,
   * hit-test with elementFromPoint, then unwrap everything.
   * Returns the character index within the block's text, or -1 if miss.
   */
  function hitTestCharIndex(block, x, y) {
    var map = getBlockTextMap(block);
    if (!map.text.length) return -1;

    // Build a flat list of char spans by replacing each text node
    var charSpans = [];
    var originalNodes = []; // track originals for restoration

    for (var t = 0; t < map.textNodes.length; t++) {
      var entry = map.textNodes[t];
      var textNode = entry.node;
      var text = textNode.nodeValue || "";
      if (!text.length) continue;

      var frag = document.createDocumentFragment();
      for (var i = 0; i < text.length; i++) {
        var span = document.createElement("span");
        span.textContent = text[i];
        span.setAttribute("data-wr-idx", String(entry.startOffset + i));
        frag.appendChild(span);
        charSpans.push(span);
      }

      originalNodes.push({ parent: textNode.parentNode, textNode: textNode, frag: frag });
      textNode.parentNode.replaceChild(frag, textNode);
    }

    // Hit test
    var hitEl = document.elementFromPoint(x, y);
    var charIndex = -1;
    if (hitEl && hitEl.hasAttribute && hitEl.hasAttribute("data-wr-idx")) {
      charIndex = parseInt(hitEl.getAttribute("data-wr-idx"), 10);
    }

    // Restore: replace char spans back with original text nodes
    // We need to do this carefully since the frag was consumed.
    // Collect spans by parent and replace them back with text nodes.
    for (var r = originalNodes.length - 1; r >= 0; r--) {
      var info = originalNodes[r];
      // Find the first char span that belongs to this text node
      // They're consecutive siblings in the parent
      var parent = info.parent;
      var textVal = info.textNode.nodeValue || "";
      var startIdx = -1;

      // Find where our spans are in the parent's children
      for (var c = 0; c < parent.childNodes.length; c++) {
        var child = parent.childNodes[c];
        if (child.hasAttribute && child.hasAttribute("data-wr-idx")) {
          var idx = parseInt(child.getAttribute("data-wr-idx"), 10);
          // Check if this is the first char of our text node
          var expectedStart = 0;
          for (var s = 0; s < map.textNodes.length; s++) {
            if (map.textNodes[s].node === info.textNode) {
              expectedStart = map.textNodes[s].startOffset;
              break;
            }
          }
          if (idx === expectedStart) {
            startIdx = c;
            break;
          }
        }
      }

      if (startIdx >= 0) {
        // Remove the char spans and insert original text node
        var count = textVal.length;
        for (var rem = 0; rem < count; rem++) {
          if (startIdx < parent.childNodes.length) {
            parent.removeChild(parent.childNodes[startIdx]);
          }
        }
        if (startIdx < parent.childNodes.length) {
          parent.insertBefore(info.textNode, parent.childNodes[startIdx]);
        } else {
          parent.appendChild(info.textNode);
        }
      }
    }

    return charIndex;
  }

  // ---------- Highlight management ----------

  function clearHighlight() {
    var existing = document.getElementById(HIGHLIGHT_ID);
    if (!existing) return;
    // Move children back to parent, then remove the span
    var parent = existing.parentNode;
    while (existing.firstChild) {
      parent.insertBefore(existing.firstChild, existing);
    }
    parent.removeChild(existing);
    // Normalize to merge adjacent text nodes
    parent.normalize();
  }

  function setActiveBlock(block) {
    // Clear old active block marker
    var old = document.getElementById(ACTIVE_BLOCK_ID);
    if (old && old !== block) {
      old.removeAttribute("id");
    }
    if (block) {
      block.id = ACTIVE_BLOCK_ID;
    }
  }

  function getActiveBlock() {
    return document.getElementById(ACTIVE_BLOCK_ID);
  }

  /**
   * Highlight a range of characters in the active block.
   * start: char index in block text
   * length: number of chars to highlight
   * Returns { rects: [...] } or null
   */
  function highlightRange(start, length) {
    clearHighlight();

    var block = getActiveBlock();
    if (!block) return null;

    var map = getBlockTextMap(block);
    var end = start + length;
    if (end > map.text.length) return null;

    // Find the Range covering [start, end) in the text nodes
    var range = document.createRange();
    var foundStart = false;
    var foundEnd = false;

    for (var i = 0; i < map.textNodes.length; i++) {
      var entry = map.textNodes[i];
      var nodeText = entry.node.nodeValue || "";
      var nodeStart = entry.startOffset;
      var nodeEnd = nodeStart + nodeText.length;

      if (!foundStart && start >= nodeStart && start < nodeEnd) {
        range.setStart(entry.node, start - nodeStart);
        foundStart = true;
      }
      if (foundStart && end >= nodeStart && end <= nodeEnd) {
        range.setEnd(entry.node, end - nodeStart);
        foundEnd = true;
        break;
      }
    }

    if (!foundStart || !foundEnd) return null;

    // Wrap the range in a highlight span
    var highlightSpan = document.createElement("span");
    highlightSpan.id = HIGHLIGHT_ID;
    highlightSpan.style.setProperty("background-color", "rgba(61, 158, 255, 0.3)", "important");
    highlightSpan.style.setProperty("border-radius", "4px", "important");

    try {
      range.surroundContents(highlightSpan);
    } catch (e) {
      // surroundContents fails if range crosses element boundaries.
      // Fallback: extract and wrap.
      var contents = range.extractContents();
      highlightSpan.appendChild(contents);
      range.insertNode(highlightSpan);
    }

    // Return bounding rects
    var rects = [];
    var domRects = highlightSpan.getClientRects();
    for (var r = 0; r < domRects.length; r++) {
      var rect = domRects[r];
      rects.push({ x: rect.left, y: rect.top, width: rect.width, height: rect.height });
    }

    return { rects: rects };
  }

  // ---------- Block navigation ----------

  /**
   * Collect all block-level elements that directly contain text.
   * "Directly contain text" means the block has at least one text node
   * child (possibly nested in inline elements) with non-whitespace content.
   */
  function getAllTextBlocks() {
    var candidates = document.querySelectorAll(BLOCK_SELECTOR);
    var blocks = [];
    for (var i = 0; i < candidates.length; i++) {
      var el = candidates[i];
      // Skip blocks that contain other blocks (we want leaf blocks)
      if (el.querySelector(BLOCK_SELECTOR)) continue;
      // Check it has actual text
      var text = el.textContent || "";
      if (text.trim().length > 0) {
        blocks.push(el);
      }
    }
    return blocks;
  }

  function getAdjacentBlock(direction) {
    var current = getActiveBlock();
    if (!current) return { atBoundary: direction === "prev" ? "start" : "end" };

    var blocks = getAllTextBlocks();
    var idx = -1;
    for (var i = 0; i < blocks.length; i++) {
      if (blocks[i] === current) { idx = i; break; }
    }
    if (idx < 0) return { atBoundary: direction === "prev" ? "start" : "end" };

    var nextIdx = direction === "next" ? idx + 1 : idx - 1;
    if (nextIdx < 0) return { atBoundary: "start" };
    if (nextIdx >= blocks.length) return { atBoundary: "end" };

    var newBlock = blocks[nextIdx];
    setActiveBlock(newBlock);
    var text = newBlock.textContent || "";
    var charIndex = direction === "next" ? 0 : Math.max(0, text.length - 1);

    return { blockText: text, charIndex: charIndex };
  }

  function getFirstOrLastBlock(position) {
    var blocks = getAllTextBlocks();
    if (!blocks.length) return null;

    var block = position === "first" ? blocks[0] : blocks[blocks.length - 1];
    setActiveBlock(block);
    var text = block.textContent || "";
    var charIndex = position === "first" ? 0 : Math.max(0, text.length - 1);

    return { blockText: text, charIndex: charIndex };
  }

  // ---------- Selection suppression ----------

  function addClass(cls) {
    if (!document.documentElement.classList.contains(cls)) {
      document.documentElement.classList.add(cls);
    }
  }

  function removeClass(cls) {
    document.documentElement.classList.remove(cls);
  }

  // ---------- Public API ----------

  window.WR = {
    __ready: true,

    setSelectable: function (selectable) {
      if (selectable) {
        removeClass("wr-nonselectable");
      } else {
        addClass("wr-nonselectable");
      }
    },

    /**
     * Find the block element and character index at a screen coordinate.
     * Returns { blockText, charIndex } or null.
     */
    getBlockAndCharIndexAtPoint: function (x, y) {
      x = Number(x);
      y = Number(y);

      var block = findBlockFromPoint(x, y);
      if (!block) return null;

      setActiveBlock(block);

      var charIndex = hitTestCharIndex(block, x, y);
      if (charIndex < 0) return null;

      var text = block.textContent || "";
      return { blockText: text, charIndex: charIndex };
    },

    /**
     * Highlight characters [start, start+length) in the last active block.
     * Returns { rects: [...] } or null.
     */
    highlightRangeInLastBlock: function (start, length) {
      return highlightRange(Number(start), Number(length));
    },

    /**
     * Clear the current highlight, restoring the DOM.
     */
    clearHighlight: function () {
      clearHighlight();
    },

    /**
     * Navigate to the adjacent block in document order.
     * direction: "prev" or "next"
     * Returns { blockText, charIndex } or { atBoundary: "start"|"end" }
     */
    getAdjacentBlock: function (direction) {
      return getAdjacentBlock(direction);
    },

    /**
     * Get the first or last text block in the document.
     * position: "first" or "last"
     * Returns { blockText, charIndex } or null.
     */
    getFirstOrLastBlock: function (position) {
      return getFirstOrLastBlock(position);
    }
  };
})();
