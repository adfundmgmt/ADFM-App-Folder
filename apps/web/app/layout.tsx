import type { Metadata } from "next";

import { Sidebar } from "../components/sidebar";
import "./globals.css";

export const metadata: Metadata = {
  title: "ADFM Analytics",
  description: "Internal analytics platform for AD Fund Management LP",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <div className="app-shell">
          <Sidebar />
          <main className="app-main">{children}</main>
        </div>
      </body>
    </html>
  );
}
