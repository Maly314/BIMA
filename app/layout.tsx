import type { Metadata, Viewport } from "next";
import { Geist } from "next/font/google";
import "./globals.css";

const geist = Geist({ variable: "--font-geist", subsets: ["latin"] });

export const metadata: Metadata = {
  title: {
    default: "BIMA",
    template: "%s · BIMA",
  },
  applicationName: "BIMA",
  description: "Synchronized sensor and pose data capture.",
  manifest: "/manifest.webmanifest",
  icons: {
    icon: [
      { url: "/bima-desktop.ico", sizes: "any" },
      { url: "/bima-icon-192.png", type: "image/png", sizes: "192x192" },
    ],
    shortcut: "/bima-desktop.ico",
    apple: "/apple-touch-icon.png",
  },
};

export const viewport: Viewport = {
  themeColor: "#087886",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body className={geist.variable}>{children}</body></html>;
}
