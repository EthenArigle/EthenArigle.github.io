(() => {
  const root = document.documentElement;
  const themeButton = document.querySelector('.theme-toggle');
  const updateThemeButton = () => {
    const dark = root.dataset.theme === 'dark';
    themeButton?.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
    themeButton?.setAttribute('aria-pressed', String(dark));
  };
  updateThemeButton();
  themeButton?.addEventListener('click', () => {
    const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
    root.dataset.theme = next;
    localStorage.setItem('theme', next);
    updateThemeButton();
  });

  const navButton = document.querySelector('.nav-toggle');
  const links = document.querySelector('#nav-links');
  navButton?.addEventListener('click', () => {
    const open = links?.classList.toggle('open') ?? false;
    navButton.setAttribute('aria-expanded', String(open));
  });
  links?.querySelectorAll('a').forEach((link) => link.addEventListener('click', () => {
    links.classList.remove('open');
    navButton?.setAttribute('aria-expanded', 'false');
  }));
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      links?.classList.remove('open');
      navButton?.setAttribute('aria-expanded', 'false');
    }
  });
})();
