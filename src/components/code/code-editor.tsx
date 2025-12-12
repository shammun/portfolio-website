"use client";

import { useCallback, useRef } from "react";
import Editor, { type Monaco } from "@monaco-editor/react";

interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  language: string;
  height: number;
}

export function CodeEditor({ value, onChange, language, height }: CodeEditorProps) {
  const editorRef = useRef<Parameters<NonNullable<Parameters<typeof Editor>[0]["onMount"]>>[0] | null>(null);

  const handleEditorDidMount = useCallback(
    (editor: Parameters<NonNullable<Parameters<typeof Editor>[0]["onMount"]>>[0], monaco: Monaco) => {
      editorRef.current = editor;

      // Configure Python language
      monaco.languages.setLanguageConfiguration("python", {
        comments: {
          lineComment: "#",
        },
        brackets: [
          ["{", "}"],
          ["[", "]"],
          ["(", ")"],
        ],
        autoClosingPairs: [
          { open: "{", close: "}" },
          { open: "[", close: "]" },
          { open: "(", close: ")" },
          { open: '"', close: '"' },
          { open: "'", close: "'" },
        ],
      });

      // Set editor theme
      monaco.editor.defineTheme("custom-dark", {
        base: "vs-dark",
        inherit: true,
        rules: [],
        colors: {
          "editor.background": "#1E1E1E",
          "editor.foreground": "#D4D4D4",
          "editorLineNumber.foreground": "#858585",
          "editor.selectionBackground": "#264F78",
          "editor.inactiveSelectionBackground": "#3A3D41",
        },
      });

      monaco.editor.setTheme("custom-dark");
    },
    []
  );

  return (
    <div style={{ height }}>
      <Editor
        height="100%"
        language={language}
        value={value}
        onChange={(val) => onChange(val || "")}
        onMount={handleEditorDidMount}
        options={{
          minimap: { enabled: false },
          fontSize: 14,
          fontFamily: "'JetBrains Mono', monospace",
          lineNumbers: "on",
          scrollBeyondLastLine: false,
          automaticLayout: true,
          padding: { top: 12, bottom: 12 },
          wordWrap: "on",
          tabSize: 4,
          insertSpaces: true,
          renderLineHighlight: "line",
          cursorBlinking: "smooth",
          smoothScrolling: true,
        }}
        theme="vs-dark"
      />
    </div>
  );
}
