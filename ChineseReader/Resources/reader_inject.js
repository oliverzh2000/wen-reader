/* Injected into every spine document. Provides a tiny API namespace. */
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

  // Expose a minimal toggle
  window.CR = {
    __ready: true,
    setSelectable: function (selectable) {
      if (selectable) {
        removeClass('cr-nonselectable');
      } else {
        addClass('cr-nonselectable');
      }
    }
  };

  // Ensure CSS is present (in case it wasn't injected as a separate userScript)
  // You can comment this out if you always inject reader_inject.css separately.
  const fallbackCSS =
    "html.cr-nonselectable, html.cr-nonselectable body, html.cr-nonselectable *{ -webkit-user-select:none !important; user-select:none !important; -webkit-touch-callout:none !important; }";
  injectStyleTag(fallbackCSS, "cr-nonselectable-fallback-style");
})();
