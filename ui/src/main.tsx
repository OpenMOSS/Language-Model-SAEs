import React from "react";
import ReactDOM from "react-dom/client";
import "@xyflow/react/dist/style.css";
import "./globals.css";
import { RouterProvider, createBrowserRouter } from "react-router-dom";
import { AppStateProvider } from "./contexts/AppStateContext";
import { FeaturesPage } from "@/routes/features/page";
import { RootPage } from "./routes/page";
import { AttentionHeadPage } from "./routes/attn-heads/page";
import { DictionaryPage } from "./routes/dictionaries/page";
import { ModelsPage } from "./routes/models/page";
import BookmarksPage from "./routes/bookmarks/page";
import { CircuitsPage } from "./routes/circuits/page";
import GraphDiffingPage from "./routes/graph-diffing/page";

const router = createBrowserRouter([
  {
    path: "/features",
    element: <FeaturesPage />,
  },
  {
    path: "/dictionaries",
    element: <DictionaryPage />,
  },
  {
    path: "/attn-heads",
    element: <AttentionHeadPage />,
  },
  {
    path: "/models",
    element: <ModelsPage />,
  },
  {
    path: "/bookmarks",
    element: <BookmarksPage />,
  },
  {
    path: "/circuits",
    element: <CircuitsPage />,
  },
  {
    path: "/graph-diffing",
    element: <GraphDiffingPage />,
  },
  {
    path: "/",
    element: <RootPage />,
  },
]);

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <AppStateProvider>
      <RouterProvider router={router} />
    </AppStateProvider>
  </React.StrictMode>
);
