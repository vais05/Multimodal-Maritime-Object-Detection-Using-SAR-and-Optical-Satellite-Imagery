import { ReactNode } from 'react';
import Sidebar from './Sidebar';
import { Page } from '../App';

interface Props {
  children: ReactNode;
  currentPage: Page;
  onNavigate: (page: Page) => void;
}

export default function Layout({ children, currentPage, onNavigate }: Props) {
  return (
    <div className="flex h-screen bg-slate-950 overflow-hidden">
      <Sidebar currentPage={currentPage} onNavigate={onNavigate} />
      <main className="flex-1 overflow-y-auto">
        <div className="min-h-full">
          {children}
        </div>
      </main>
    </div>
  );
}
