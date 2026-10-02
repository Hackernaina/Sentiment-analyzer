import { useState } from "react";
import Intro from "./components/Intro";
import Analyzer from "./components/Analyzer";

export default function App() {
  const [showIntro, setShowIntro] = useState(true);

  return (
    <main className="app-shell">
      <div className="aurora aurora-one" />
      <div className="aurora aurora-two" />
      {showIntro ? (
        <Intro onEnter={() => setShowIntro(false)} />
      ) : (
        <Analyzer onBack={() => setShowIntro(true)} />
      )}
    </main>
  );
}
