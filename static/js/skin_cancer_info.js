const cards = document.querySelectorAll(
  ".type-card, .abcde-card, .risk-factor"
);

cards.forEach((card, index) => {
  card.style.transitionDelay = `${index * 0.03}s`;
});

const riskTags = document.querySelectorAll(".risk-tag");

riskTags.forEach((tag) => {
  tag.addEventListener("mouseenter", () => {
    tag.style.transform = "scale(1.05)";
  });

  tag.addEventListener("mouseleave", () => {
    tag.style.transform = "scale(1)";
  });
});