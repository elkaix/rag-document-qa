import { useRef, useState } from "react";
import type { DragEvent } from "react";
import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";

const ACCEPTED_TYPES =
  ".pdf,.txt,.md,.html,.htm,.csv,.docx,.json";

interface DropzoneProps {
  onFiles: (files: File[]) => void;
}

export function Dropzone({ onFiles }: DropzoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) onFiles(files);
  }

  function handleDragOver(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(true);
  }

  function handleDragLeave(e: DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragOver(false);
  }

  function handleClick() {
    inputRef.current?.click();
  }

  function handleInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? []);
    if (files.length > 0) onFiles(files);
    // Reset input so the same file can be re-selected
    e.target.value = "";
  }

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={handleClick}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") handleClick();
      }}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl p-12 text-center transition-all duration-200 border-2 border-dashed",
        dragOver
          ? "border-[#0d74e7] bg-[#EBF3FE] shadow-[0_0_20px_rgba(13,116,231,0.1)]"
          : "border-[#D1D5DB] hover:border-[#0d74e7] hover:bg-[#EBF3FE]/50"
      )}
    >
      <Upload className="size-10 text-muted-foreground" />
      <div className="flex flex-col gap-1">
        <p className="text-sm font-medium">
          Drop files here or click to browse
        </p>
        <p className="text-xs text-muted-foreground">
          Supports PDF, TXT, MD, HTML, CSV, DOCX, JSON
        </p>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES}
        multiple
        className="hidden"
        onChange={handleInputChange}
      />
    </div>
  );
}
