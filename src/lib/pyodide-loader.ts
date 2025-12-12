"use client";

import type { PyodideInterface } from "pyodide";

let pyodideInstance: PyodideInterface | null = null;
let loadingPromise: Promise<PyodideInterface> | null = null;
const loadedPackages = new Set<string>(["numpy", "matplotlib"]);

export interface PythonOutput {
  type: "stdout" | "stderr" | "result" | "error" | "plot" | "info";
  content: string;
}

export async function loadPyodide(): Promise<PyodideInterface> {
  if (pyodideInstance) {
    return pyodideInstance;
  }

  if (loadingPromise) {
    return loadingPromise;
  }

  loadingPromise = (async () => {
    // Dynamically import pyodide
    const { loadPyodide: loadPyodideModule } = await import("pyodide");

    const pyodide = await loadPyodideModule({
      indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/",
    });

    // Load commonly used scientific packages
    await pyodide.loadPackage(["numpy", "matplotlib"]);

    // Set up matplotlib for browser output
    await pyodide.runPythonAsync(`
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import io
import base64

def get_plot_as_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='#1E1E1E', edgecolor='none')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64
    `);

    pyodideInstance = pyodide;
    return pyodide;
  })();

  return loadingPromise;
}

export async function runPython(
  code: string,
  onOutput?: (output: PythonOutput) => void
): Promise<PythonOutput[]> {
  const outputs: PythonOutput[] = [];

  try {
    const pyodide = await loadPyodide();

    // Capture stdout and stderr
    let stdout = "";
    let stderr = "";

    pyodide.setStdout({
      batched: (text: string) => {
        stdout += text + "\n";
        onOutput?.({ type: "stdout", content: text });
      },
    });

    pyodide.setStderr({
      batched: (text: string) => {
        stderr += text + "\n";
        onOutput?.({ type: "stderr", content: text });
      },
    });

    // Run the code
    const result = await pyodide.runPythonAsync(code);

    // Check if there's a matplotlib figure
    const hasPlot = await pyodide.runPythonAsync(`
import matplotlib.pyplot as plt
len(plt.get_fignums()) > 0
    `);

    if (hasPlot) {
      const plotBase64 = await pyodide.runPythonAsync("get_plot_as_base64()");
      outputs.push({ type: "plot", content: plotBase64 });
      onOutput?.({ type: "plot", content: plotBase64 });
    }

    // Add stdout if any
    if (stdout.trim()) {
      outputs.push({ type: "stdout", content: stdout.trim() });
    }

    // Add stderr if any
    if (stderr.trim()) {
      outputs.push({ type: "stderr", content: stderr.trim() });
    }

    // Add result if not undefined
    if (result !== undefined && result !== null) {
      const resultStr = String(result);
      if (resultStr && resultStr !== "undefined" && resultStr !== "None") {
        outputs.push({ type: "result", content: resultStr });
        onOutput?.({ type: "result", content: resultStr });
      }
    }

    return outputs;
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : String(error);
    outputs.push({ type: "error", content: errorMessage });
    onOutput?.({ type: "error", content: errorMessage });
    return outputs;
  }
}

export function isPyodideLoaded(): boolean {
  return pyodideInstance !== null;
}

// Load additional packages on demand
export async function loadPackages(
  packages: string[],
  onProgress?: (message: string) => void
): Promise<void> {
  const pyodide = await loadPyodide();

  const packagesToLoad = packages.filter((pkg) => !loadedPackages.has(pkg));

  if (packagesToLoad.length === 0) {
    return;
  }

  onProgress?.(`Loading packages: ${packagesToLoad.join(", ")}...`);

  try {
    await pyodide.loadPackage(packagesToLoad);
    packagesToLoad.forEach((pkg) => loadedPackages.add(pkg));
    onProgress?.(`Packages loaded: ${packagesToLoad.join(", ")}`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to load packages: ${message}`);
  }
}

// Get list of loaded packages
export function getLoadedPackages(): string[] {
  return Array.from(loadedPackages);
}

// Parse import statements from code to auto-detect needed packages
export function detectRequiredPackages(code: string): string[] {
  const packageMap: Record<string, string> = {
    scipy: "scipy",
    sklearn: "scikit-learn",
    pandas: "pandas",
    torch: "torch", // Note: torch is not available in Pyodide, this is for info
    sympy: "sympy",
    networkx: "networkx",
    pillow: "Pillow",
    pil: "Pillow",
  };

  const importRegex = /^(?:import|from)\s+(\w+)/gm;
  const detectedPackages: Set<string> = new Set();
  let match;

  while ((match = importRegex.exec(code)) !== null) {
    const moduleName = match[1].toLowerCase();
    if (packageMap[moduleName] && !loadedPackages.has(packageMap[moduleName])) {
      detectedPackages.add(packageMap[moduleName]);
    }
  }

  return Array.from(detectedPackages);
}
