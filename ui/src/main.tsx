import React from "react";
import ReactDOM from "react-dom/client";
import "./globals.css";
import { RouterProvider, createBrowserRouter } from "react-router-dom";
import { FeaturesPage } from "@/routes/features/page";
import { RootPage } from "./routes/page";
import { AttentionHeadPage } from "./routes/attn-heads/page";

const router = createBrowserRouter([
  {
    path: "/features",
    element: <FeaturesPage />,
  },
  {
    path: "/attn-head",
    element: <AttentionHeadPage />,
  },
  {
    path: "/",
    element: <RootPage />,
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
);
