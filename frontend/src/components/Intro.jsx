export default function Intro({ onEnter }) {
  return (
    <section className="intro-page">
      <div className="floating-ring ring-a" />
      <div className="floating-ring ring-b" />

      <div className="brand-mark">
        <span>⚡</span>
      </div>

      <p className="eyebrow">AI • EMOTION • LANGUAGE</p>
      <h1>
        <span className="gradient-text">smh</span> cheerful
      </h1>

      <p className="hero-copy">
        A human sentiment analyzer that looks beyond simple positive and
        negative labels to uncover the emotional signals hidden in your words.
      </p>

      <div className="intro-pills">
        <span>15 emotions</span>
        <span>Unsupervised ML</span>
        <span>Semantic AI</span>
      </div>

      <button className="primary-btn" onClick={onEnter}>
        Analyze my emotions
        <span>→</span>
      </button>

      <p className="tiny-note">
        Built with React + FastAPI + Hugging Face embeddings
      </p>
    </section>
  );
}
