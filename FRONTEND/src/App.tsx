import { useState, useEffect } from 'react';
import { useAuth } from './hooks/useAuth';
import AuthPage from './pages/AuthPage';
import Layout from './components/Layout';
import DashboardPage from './pages/DashboardPage';
import PipelinePage from './pages/PipelinePage';
import DetectionPage from './pages/DetectionPage';
import AccountPage from './pages/AccountPage';
import ArchitecturePage from './pages/ArchitecturePage';
import PhasePage from './pages/PhasePage';

export type Page = 'dashboard' | 'pipeline' | 'detection' | 'account' | 'architecture' | `phase-${number}`;

export default function App() {
  const { user, loading } = useAuth();
  const [currentPage, setCurrentPage] = useState<Page>('dashboard');

  useEffect(() => {
    if (!user) setCurrentPage('dashboard');
  }, [user]);

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <div className="w-16 h-16 border-4 border-sky-500 border-t-transparent rounded-full animate-spin" />
          <p className="text-slate-400 text-sm tracking-widest uppercase">Initialising</p>
        </div>
      </div>
    );
  }

  if (!user) return <AuthPage />;

  const renderPage = () => {
    if (currentPage === 'dashboard') return <DashboardPage onNavigate={setCurrentPage} />;
    if (currentPage === 'pipeline') return <PipelinePage onNavigate={setCurrentPage} />;
    if (currentPage === 'detection') return <DetectionPage />;
    if (currentPage === 'account') return <AccountPage />;
    if (currentPage === 'architecture') return <ArchitecturePage />;
    if (currentPage.startsWith('phase-')) {
      const phase = parseInt(currentPage.split('-')[1]);
      return <PhasePage phase={phase} onBack={() => setCurrentPage('pipeline')} />;
    }
    return <DashboardPage onNavigate={setCurrentPage} />;
  };

  return (
    <Layout currentPage={currentPage} onNavigate={setCurrentPage}>
      {renderPage()}
    </Layout>
  );
}
