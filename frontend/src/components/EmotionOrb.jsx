const icons = {
  joy: "☀️",
  sadness: "🌧️",
  anger: "🔥",
  fear: "🌫️",
  surprise: "⚡",
  disgust: "🫥",
  love: "💗",
  gratitude: "🌿",
  guilt: "🪞",
  shame: "🫣",
  anxiety: "〰️",
  hope: "🌱",
  frustration: "💥",
  loneliness: "🌙",
  disappointment: "☁️",
};

export default function EmotionOrb({ emotion }) {
  return (
    <div className="emotion-orb">
      <div className="orb-core">
        <span>{icons[emotion] || "✨"}</span>
      </div>
      <div className="orb-ring ring-one" />
      <div className="orb-ring ring-two" />
    </div>
  );
}
