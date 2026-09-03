import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, MemoryRouter, Navigate, Route, Routes, useLocation, useParams } from "react-router-dom";

import { AppShell } from "./AppShell";
import { AuthProvider, RequireAuth } from "../auth/AuthProvider";
import { AttachmentsPage } from "../pages/AttachmentsPage";
import { ChannelsPage } from "../pages/ChannelsPage";
import { CollaborationPage } from "../pages/CollaborationPage";
import { ConfigPage } from "../pages/ConfigPage";
import { LoginPage } from "../pages/LoginPage";
import { LogsPage } from "../pages/LogsPage";
import { MainAgentPage } from "../pages/MainAgentPage";
import { McpPage } from "../pages/McpPage";
import { MemoryPage } from "../pages/MemoryPage";
import { ModelsPage } from "../pages/ModelsPage";
import { OpenClawPage } from "../pages/OpenClawPage";
import { ModuleHubPage } from "../pages/ModuleHubPage";
import { RunDetailPage } from "../pages/RunDetailPage";
import { RunsPage } from "../pages/RunsPage";
import { SchedulesPage } from "../pages/SchedulesPage";
import { SetupPage } from "../pages/SetupPage";
import { SkillsPage } from "../pages/SkillsPage";
import { UsersPage } from "../pages/UsersPage";
import { MODULE_GROUPS } from "./navigation";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false },
    mutations: { retry: false },
  },
});

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/setup" element={<SetupPage />} />
      <Route
        path="/"
        element={
          <RequireAuth>
            <AppShell />
          </RequireAuth>
        }
      >
        <Route index element={<RunsPage />} />
        {MODULE_GROUPS.map((group) => (
          <Route key={group.id} path={group.to.slice(1)} element={<ModuleHubPage group={group} />} />
        ))}
        <Route path="runs/:runId" element={<RunDetailPage />} />
        <Route path="evolution" element={<Navigate to="/" replace />} />
        <Route path="config" element={<ConfigPage />} />
        <Route path="main-agent" element={<MainAgentPage />} />
        <Route path="models" element={<ModelsPage />} />
        <Route path="openclaw" element={<OpenClawPage />} />
        <Route path="multimedia" element={<Navigate to="/models" replace />} />
        <Route path="attachments" element={<AttachmentsPage />} />
        <Route path="collaboration" element={<CollaborationPage />} />
        <Route path="agents" element={<Navigate to="/collaboration?section=roles" replace />} />
        <Route path="workflows" element={<Navigate to="/collaboration?section=workflows" replace />} />
        <Route path="schedules" element={<SchedulesPage />} />
        <Route path="skills" element={<SkillsPage />} />
        <Route path="mcp" element={<McpPage />} />
        <Route path="channels" element={<ChannelsPage />} />
        <Route path="memory" element={<MemoryPage />} />
        <Route path="hermes" element={<HermesLegacyRedirect />} />
        <Route path="hermes/:insightId" element={<HermesLegacyRedirect />} />
        <Route path="users" element={<UsersPage />} />
        <Route path="logs" element={<LogsPage />} />
        <Route path="logs/:module" element={<LogsPage />} />
        <Route path="audit" element={<Navigate to="/logs/audit" replace />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}

function HermesLegacyRedirect() {
  const location = useLocation();
  const { insightId } = useParams();
  const params = new URLSearchParams(location.search);
  params.set("source", "hermes");
  if (insightId) params.set("insight", insightId);
  return <Navigate to={`/memory?${params.toString()}`} replace />;
}

export function AppRouter() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AuthProvider>
          <AppRoutes />
        </AuthProvider>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export function TestApp({ initialPath = "/" }: { initialPath?: string }) {
  const testClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return (
    <QueryClientProvider client={testClient}>
      <MemoryRouter initialEntries={[initialPath]}>
        <AuthProvider>
          <AppRoutes />
        </AuthProvider>
      </MemoryRouter>
    </QueryClientProvider>
  );
}
