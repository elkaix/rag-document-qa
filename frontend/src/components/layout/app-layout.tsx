import { Outlet } from "react-router";
import { Sidebar } from "./sidebar";

export function AppLayout() {
  return (
    <div className="grid h-screen w-screen grid-cols-[auto_1fr] overflow-hidden">
      <Sidebar />
      <main className="overflow-hidden dot-pattern">
        <Outlet />
      </main>
    </div>
  );
}
