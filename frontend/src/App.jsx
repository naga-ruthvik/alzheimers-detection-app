import { useState } from 'react';
import Navbar from './components/Navbar';
import BackgroundEffects from './components/BackgroundEffects';
import LandingPage from './pages/LandingPage';
import TryPage from './pages/TryPage';
import './App.css';

function App() {
  const [currentPage, setCurrentPage] = useState('landing');

  const navigate = (page) => setCurrentPage(page);

  return (
    <div className="flex flex-col min-h-screen bg-transparent text-slate-50 relative selection:bg-brand-500/30 selection:text-brand-400">
      <BackgroundEffects />
      <Navbar currentPage={currentPage} onNavigate={navigate} />

      {currentPage === 'landing' ? (
        <LandingPage onNavigate={navigate} />
      ) : (
        <TryPage onNavigate={navigate} />
      )}
    </div>
  );
}

export default App;