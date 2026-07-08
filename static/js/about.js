const noveltyCards = document.querySelectorAll(".novelty-card");
const workflowBoxes = document.querySelectorAll(".workflow-box");

noveltyCards.forEach((card, index) => {
  card.style.transitionDelay = `${index * 0.03}s`;
});

workflowBoxes.forEach((box, index) => {
  box.style.transitionDelay = `${index * 0.05}s`;
});