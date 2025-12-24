import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Introduction from "./pages/docs/Introduction";
import QuickStart from "./pages/docs/QuickStart";
import Architecture from "./pages/docs/Architecture";
import Configuration from "./pages/docs/Configuration";
import Training from "./pages/docs/Training";
import Sampling from "./pages/docs/Sampling";
import ApiModel from "./pages/docs/api/ApiModel";
import ApiAttention from "./pages/docs/api/ApiAttention";
import ApiBlock from "./pages/docs/api/ApiBlock";
import ApiEmbedding from "./pages/docs/api/ApiEmbedding";
import ApiUtils from "./pages/docs/api/ApiUtils";
import ApiTrain from "./pages/docs/api/ApiTrain";
import ApiSample from "./pages/docs/api/ApiSample";
import License from "./pages/legal/License";
import CodeOfConduct from "./pages/legal/CodeOfConduct";
import Security from "./pages/legal/Security";
import Citation from "./pages/legal/Citation";
import Contributing from "./pages/Contributing";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/docs" element={<Introduction />} />
          <Route path="/docs/quickstart" element={<QuickStart />} />
          <Route path="/docs/architecture" element={<Architecture />} />
          <Route path="/docs/config" element={<Configuration />} />
          <Route path="/docs/training" element={<Training />} />
          <Route path="/docs/sampling" element={<Sampling />} />
          <Route path="/docs/api/model" element={<ApiModel />} />
          <Route path="/docs/api/attention" element={<ApiAttention />} />
          <Route path="/docs/api/block" element={<ApiBlock />} />
          <Route path="/docs/api/embedding" element={<ApiEmbedding />} />
          <Route path="/docs/api/utils" element={<ApiUtils />} />
          <Route path="/docs/api/train" element={<ApiTrain />} />
          <Route path="/docs/api/sample" element={<ApiSample />} />
          <Route path="/legal/license" element={<License />} />
          <Route path="/legal/code-of-conduct" element={<CodeOfConduct />} />
          <Route path="/legal/security" element={<Security />} />
          <Route path="/legal/citation" element={<Citation />} />
          <Route path="/contribute" element={<Contributing />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
