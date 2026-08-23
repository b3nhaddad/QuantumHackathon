const slides = [...document.querySelectorAll('.slide')];
const previous = document.querySelector('#prev');
const next = document.querySelector('#next');
const currentLabel = document.querySelector('#currentSlide');
const totalLabel = document.querySelector('#totalSlides');
const progress = document.querySelector('#progressBar');

let current = 0;

function clamp(index) {
  return Math.max(0, Math.min(slides.length - 1, index));
}

function render(index, updateHash = true) {
  current = clamp(index);
  slides.forEach((slide, i) => {
    const active = i === current;
    slide.classList.toggle('is-active', active);
    slide.setAttribute('aria-hidden', String(!active));
  });

  currentLabel.textContent = String(current + 1);
  totalLabel.textContent = String(slides.length);
  progress.style.width = `${((current + 1) / slides.length) * 100}%`;
  previous.disabled = current === 0;
  next.disabled = current === slides.length - 1;
  document.title = `${slides[current].dataset.title} — Qupacabrathon 2026`;

  if (updateHash) history.replaceState(null, '', `#${current + 1}`);
}

function move(delta) { render(current + delta); }

previous.addEventListener('click', () => move(-1));
next.addEventListener('click', () => move(1));

document.addEventListener('keydown', event => {
  if (['ArrowRight', 'PageDown', 'Enter', ' '].includes(event.key)) {
    event.preventDefault();
    move(1);
  }
  if (['ArrowLeft', 'PageUp', 'Backspace'].includes(event.key)) {
    event.preventDefault();
    move(-1);
  }
  if (event.key === 'Home') render(0);
  if (event.key === 'End') render(slides.length - 1);
});

document.querySelector('.deck').addEventListener('click', event => {
  if (event.target.closest('button, a')) return;
  move(event.clientX < window.innerWidth / 2 ? -1 : 1);
});

const initial = Number.parseInt(location.hash.slice(1), 10);
render(Number.isFinite(initial) ? initial - 1 : 0, false);
