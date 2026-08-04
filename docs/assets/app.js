(() => {
  const root = document.documentElement;
  const buttons = [...document.querySelectorAll('[data-set-language]')];
  const metadata = {
    de: {
      title: 'hostrada4py — HOSTRADA- und CERRA-Klimadaten für Python',
      description: 'hostrada4py erschließt HOSTRADA- und CERRA-Klimadaten für Zeitreihen, Klimakarten, Routing und Simulationen in Deutschland und Europa.',
      imageAlt: 'HOSTRADA-Darstellung der Übertemperatur im Stadtgebiet Berlin am 2. August 2025 um 03:00 Uhr'
    },
    en: {
      title: 'hostrada4py — HOSTRADA and CERRA climate data for Python',
      description: 'hostrada4py provides Python access to HOSTRADA and CERRA data for time series, climate maps, routing and simulations across Germany and Europe.',
      imageAlt: 'HOSTRADA map of temperature excess across Berlin on 2 August 2025 at 3:00 a.m.'
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
    document.querySelectorAll('[data-alt-de][data-alt-en]').forEach((image) => {
      image.alt = image.dataset[language === 'de' ? 'altDe' : 'altEn'];
    });
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
