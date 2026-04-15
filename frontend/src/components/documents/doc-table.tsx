import { useState } from "react";
import { ChevronDown, ChevronRight, Trash2 } from "lucide-react";
import { toast } from "sonner";
import type { DocumentInfo } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogClose,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useDeleteDocument } from "@/hooks/use-documents";
import { ChunkViewer } from "./chunk-viewer";

interface DocTableProps {
  documents: DocumentInfo[];
}

function formatSize(bytes: number | null): string {
  if (bytes == null) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(dateStr: string): string {
  return new Date(dateStr).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function DocTable({ documents }: DocTableProps) {
  const [filter, setFilter] = useState("");
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [deleteTarget, setDeleteTarget] = useState<DocumentInfo | null>(null);
  const deleteMutation = useDeleteDocument();

  const filtered = filter
    ? documents.filter((d) =>
        d.filename.toLowerCase().includes(filter.toLowerCase())
      )
    : documents;

  function handleDelete() {
    if (!deleteTarget) return;
    deleteMutation.mutate(deleteTarget.doc_id, {
      onSuccess: () => {
        toast.success(`Deleted ${deleteTarget.filename}`);
        setDeleteTarget(null);
        if (expandedId === deleteTarget.doc_id) setExpandedId(null);
      },
      onError: (err) => {
        toast.error(
          `Failed to delete: ${err instanceof Error ? err.message : "Unknown error"}`
        );
      },
    });
  }

  return (
    <div className="flex flex-col gap-4">
      <Input
        placeholder="Filter documents..."
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        className="max-w-xs"
      />

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-10" />
            <TableHead>Filename</TableHead>
            <TableHead>Type</TableHead>
            <TableHead>Size</TableHead>
            <TableHead>Chunks</TableHead>
            <TableHead>Date</TableHead>
            <TableHead className="w-10" />
          </TableRow>
        </TableHeader>
        <TableBody>
          {filtered.map((doc) => {
            const isExpanded = expandedId === doc.doc_id;
            return (
              <DocRow
                key={doc.doc_id}
                doc={doc}
                isExpanded={isExpanded}
                onToggle={() =>
                  setExpandedId(isExpanded ? null : doc.doc_id)
                }
                onDelete={() => setDeleteTarget(doc)}
              />
            );
          })}
          {filtered.length === 0 && (
            <TableRow>
              <TableCell colSpan={7} className="py-8 text-center text-muted-foreground">
                No documents match your filter.
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>

      {/* Delete confirmation dialog */}
      <Dialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete Document</DialogTitle>
            <DialogDescription>
              Are you sure you want to delete{" "}
              <strong>{deleteTarget?.filename}</strong>? This will remove the
              document and all its chunks from the collection. This action cannot
              be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <DialogClose render={<Button variant="outline" />}>
              Cancel
            </DialogClose>
            <Button
              variant="destructive"
              onClick={handleDelete}
              disabled={deleteMutation.isPending}
            >
              {deleteMutation.isPending ? "Deleting..." : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}

function DocRow({
  doc,
  isExpanded,
  onToggle,
  onDelete,
}: {
  doc: DocumentInfo;
  isExpanded: boolean;
  onToggle: () => void;
  onDelete: () => void;
}) {
  return (
    <>
      <TableRow>
        <TableCell>
          <Button variant="ghost" size="icon-xs" onClick={onToggle}>
            {isExpanded ? (
              <ChevronDown className="size-3.5" />
            ) : (
              <ChevronRight className="size-3.5" />
            )}
          </Button>
        </TableCell>
        <TableCell className="font-medium">{doc.filename}</TableCell>
        <TableCell>
          <Badge variant="outline">{doc.file_type ?? "—"}</Badge>
        </TableCell>
        <TableCell>{formatSize(doc.file_size_bytes)}</TableCell>
        <TableCell>{doc.chunks}</TableCell>
        <TableCell>{formatDate(doc.upload_date)}</TableCell>
        <TableCell>
          <Button
            variant="ghost"
            size="icon-xs"
            onClick={(e) => {
              e.stopPropagation();
              onDelete();
            }}
          >
            <Trash2 className="size-3.5 text-destructive" />
          </Button>
        </TableCell>
      </TableRow>
      {isExpanded && (
        <TableRow>
          <TableCell colSpan={7} className="p-0 whitespace-normal">
            <ChunkViewer docId={doc.doc_id} />
          </TableCell>
        </TableRow>
      )}
    </>
  );
}
