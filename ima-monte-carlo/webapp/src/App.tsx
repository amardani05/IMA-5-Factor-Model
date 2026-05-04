import { Link, NavLink, Route, Routes } from "react-router-dom";
import { PitchListPage } from "@/routes/PitchListPage";
import { PitchDetailPage } from "@/routes/PitchDetailPage";
import { ComparePage } from "@/routes/ComparePage";
import { AboutPage } from "@/routes/AboutPage";

export function App() {
  return (
    <div className="app-shell">
      <header className="nav">
        <div>
          <Link to="/" style={{ color: "inherit" }}>
            <h1 style={{ display: "inline" }}>IMA Monte Carlo</h1>
          </Link>
          <span className="subtitle">Pitch analysis dashboard</span>
        </div>
        <nav>
          <NavLink to="/" end>
            Pitches
          </NavLink>
          <NavLink to="/compare">Compare</NavLink>
          <NavLink to="/about">About</NavLink>
        </nav>
      </header>
      <main style={{ flex: 1 }}>
        <Routes>
          <Route path="/" element={<PitchListPage />} />
          <Route path="/pitch/:id" element={<PitchDetailPage />} />
          <Route path="/compare" element={<ComparePage />} />
          <Route path="/about" element={<AboutPage />} />
          <Route
            path="*"
            element={
              <div className="container empty-state">
                <h2>Not found</h2>
                <Link to="/">Back to pitches</Link>
              </div>
            }
          />
        </Routes>
      </main>
      <footer className="footer">
        Illinois Investment Management Academy • Monte Carlo Pitch Dashboard
      </footer>
    </div>
  );
}
