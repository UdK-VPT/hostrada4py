(() => {
  const root = document.documentElement;
  const buttons = [...document.querySelectorAll('[data-set-language]')];
  const metadata = {
    de: {
      title: 'hostrada4py — HOSTRADA-Klimadaten für Python',
      description: 'hostrada4py erschließt die hochaufgelösten stündlichen HOSTRADA-Klimadaten des Deutschen Wetterdienstes für Python-Workflows, Karten und Simulationen.',
      imageAlt: 'HOSTRADA-Karte der städtischen Wärmeinselintensität in Berlin'
    },
    en: {
      title: 'hostrada4py — HOSTRADA climate data for Python',
      description: 'hostrada4py makes the German Weather Service’s high-resolution hourly HOSTRADA climate data accessible for Python workflows, maps and simulations.',
      imageAlt: 'HOSTRADA map of urban heat island intensity in Berlin'
    }
  };

  function setLanguage(language, persist = true) {
    if (!metadata[language]) return;
    root.dataset.lang = language;
    root.lang = language;
    document.title = metadata[language].title;
    document.querySelector('meta[name="description"]').setAttribute('content', metadata[language].description);
    const heroImage = document.querySelector('.image-card img');
    if (heroImage) heroImage.alt = metadata[language].imageAlt;
    buttons.forEach((button) => {
      button.setAttribute('aria-pressed', String(button.dataset.setLanguage === language));
    });
    if (persist) {
      try { localStorage.setItem('hostrada4py-language', language); } catch (_) {}
    }
  }

  buttons.forEach((button) => {
    button.addEventListener('click', () => setLanguage(button.dataset.setLanguage));
  });

  let initialLanguage = 'de';
  try {
    const saved = localStorage.getItem('hostrada4py-language');
    if (saved && metadata[saved]) initialLanguage = saved;
    else if (navigator.language?.toLowerCase().startsWith('en')) initialLanguage = 'en';
  } catch (_) {}
  setLanguage(initialLanguage, false);
})();
