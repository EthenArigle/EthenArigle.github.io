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

  const emailButton = document.querySelector('.email-copy');
  const emailStatus = document.querySelector('#email-copy-status');
  const copyText = async (value) => {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(value);
      return;
    }
    const field = document.createElement('textarea');
    field.value = value;
    field.setAttribute('readonly', '');
    field.style.position = 'fixed';
    field.style.opacity = '0';
    document.body.appendChild(field);
    field.select();
    const copied = document.execCommand('copy');
    field.remove();
    if (!copied) throw new Error('Copy command was unavailable');
  };
  emailButton?.addEventListener('click', async () => {
    const email = emailButton.dataset.email;
    if (!email) return;
    try {
      await copyText(email);
      emailButton.textContent = 'Copied!';
      if (emailStatus) emailStatus.textContent = `Email address copied: ${email}`;
      window.setTimeout(() => {
        emailButton.textContent = 'Email';
        if (emailStatus) emailStatus.textContent = '';
      }, 1800);
    } catch (_error) {
      emailButton.textContent = email;
      if (emailStatus) emailStatus.textContent = `Copy failed. Email address: ${email}`;
    }
  });
})();
