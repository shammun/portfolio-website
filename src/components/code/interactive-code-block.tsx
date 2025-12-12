"use client";

import { useState, useEffect } from "react";
import dynamic from "next/dynamic";
import { Play, RotateCcw, Copy, Check, Download, Loader2, Square } from "lucide-react";
import { cn } from "@/lib/utils";
import { runPython, loadPyodide, isPyodideLoaded, type PythonOutput } from "@/lib/pyodide-loader";

// Dynamically import the code editor to avoid SSR issues with Monaco
const CodeEditor = dynamic(
  () => import("./code-editor").then((mod) => mod.CodeEditor),
  {
    ssr: false,
    loading: () => (
      <div className="h-[200px] bg-[#1E1E1E] flex items-center justify-center text-gray-400">
        <Loader2 className="h-5 w-5 animate-spin mr-2" />
        Loading editor...
      </div>
    ),
  }
);

interface InteractiveCodeBlockProps {
  initialCode: string;
  title?: string;
  language?: string;
  className?: string;
}

export function InteractiveCodeBlock({
  initialCode,
  title,
  language = "python",
  className,
}: InteractiveCodeBlockProps) {
  const [code, setCode] = useState(initialCode);
  const [outputs, setOutputs] = useState<PythonOutput[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [pyodideReady, setPyodideReady] = useState(false);
  const [copied, setCopied] = useState(false);

  // Check if Pyodide is already loaded
  useEffect(() => {
    setPyodideReady(isPyodideLoaded());
  }, []);

  // Pre-load Pyodide when component mounts
  useEffect(() => {
    if (!pyodideReady) {
      setIsLoading(true);
      loadPyodide()
        .then(() => {
          setPyodideReady(true);
          setIsLoading(false);
        })
        .catch((error) => {
          console.error("Failed to load Pyodide:", error);
          setIsLoading(false);
        });
    }
  }, [pyodideReady]);

  const handleRun = async () => {
    if (isRunning || !pyodideReady) return;

    setIsRunning(true);
    setOutputs([]);

    try {
      const results = await runPython(code, (output) => {
        setOutputs((prev) => [...prev, output]);
      });

      if (results.length === 0) {
        setOutputs([{ type: "stdout", content: "Code executed successfully (no output)" }]);
      }
    } catch (error) {
      setOutputs([
        {
          type: "error",
          content: error instanceof Error ? error.message : "Unknown error occurred",
        },
      ]);
    } finally {
      setIsRunning(false);
    }
  };

  const handleStop = () => {
    setIsRunning(false);
  };

  const handleReset = () => {
    setCode(initialCode);
    setOutputs([]);
  };

  const handleCopy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([code], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = title ? `${title.replace(/\s+/g, "_")}.py` : "code.py";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const lineCount = code.split("\n").length;
  const editorHeight = Math.min(Math.max(lineCount * 20 + 20, 100), 400);

  return (
    <div className={cn("my-6 rounded-lg border border-border overflow-hidden bg-[#1E1E1E]", className)}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-[#252526] border-b border-border">
        <div className="flex items-center gap-3">
          {title && <span className="text-sm font-medium text-gray-300">{title}</span>}
          <span className="text-xs text-gray-500 uppercase">{language}</span>
        </div>

        <div className="flex items-center gap-1">
          {/* Loading indicator */}
          {isLoading && (
            <div className="flex items-center gap-2 text-xs text-gray-400 mr-2">
              <Loader2 className="h-3 w-3 animate-spin" />
              Loading Python...
            </div>
          )}

          {/* Copy button */}
          <button
            onClick={handleCopy}
            className="p-2 rounded hover:bg-gray-700 text-gray-400 hover:text-gray-200 transition-colors"
            title="Copy code"
          >
            {copied ? <Check className="h-4 w-4 text-green-400" /> : <Copy className="h-4 w-4" />}
          </button>

          {/* Download button */}
          <button
            onClick={handleDownload}
            className="p-2 rounded hover:bg-gray-700 text-gray-400 hover:text-gray-200 transition-colors"
            title="Download code"
          >
            <Download className="h-4 w-4" />
          </button>

          {/* Reset button */}
          <button
            onClick={handleReset}
            className="p-2 rounded hover:bg-gray-700 text-gray-400 hover:text-gray-200 transition-colors"
            title="Reset code"
          >
            <RotateCcw className="h-4 w-4" />
          </button>

          {/* Run/Stop button */}
          {isRunning ? (
            <button
              onClick={handleStop}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-red-600 hover:bg-red-700 text-white text-sm font-medium transition-colors"
            >
              <Square className="h-3.5 w-3.5" />
              Stop
            </button>
          ) : (
            <button
              onClick={handleRun}
              disabled={!pyodideReady || isLoading}
              className={cn(
                "flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium transition-colors",
                pyodideReady
                  ? "bg-primary hover:bg-primary/90 text-white"
                  : "bg-gray-600 text-gray-400 cursor-not-allowed"
              )}
            >
              <Play className="h-3.5 w-3.5" />
              Run
            </button>
          )}
        </div>
      </div>

      {/* Editor */}
      <CodeEditor
        value={code}
        onChange={setCode}
        language={language}
        height={editorHeight}
      />

      {/* Output Panel */}
      {(outputs.length > 0 || isRunning) && (
        <div className="border-t border-border">
          <div className="px-4 py-2 bg-[#252526] border-b border-border">
            <span className="text-sm font-medium text-gray-300">Output</span>
          </div>
          <div className="p-4 max-h-96 overflow-auto bg-[#1E1E1E]">
            {isRunning && outputs.length === 0 && (
              <div className="flex items-center gap-2 text-gray-400">
                <Loader2 className="h-4 w-4 animate-spin" />
                Running...
              </div>
            )}

            {outputs.map((output, index) => (
              <div key={index} className="mb-2">
                {output.type === "plot" ? (
                  <img
                    src={`data:image/png;base64,${output.content}`}
                    alt="Plot output"
                    className="max-w-full rounded"
                  />
                ) : (
                  <pre
                    className={cn(
                      "text-sm font-mono whitespace-pre-wrap",
                      output.type === "error" && "text-red-400",
                      output.type === "stderr" && "text-yellow-400",
                      output.type === "stdout" && "text-gray-300",
                      output.type === "result" && "text-green-400",
                      output.type === "info" && "text-blue-400"
                    )}
                  >
                    {output.type === "error" && "Error: "}
                    {output.type === "info" && "Info: "}
                    {output.content}
                  </pre>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
