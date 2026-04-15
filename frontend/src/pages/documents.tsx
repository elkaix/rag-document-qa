import { FolderOpen } from "lucide-react";
import { Skeleton } from "@/components/ui/skeleton";
import { DocStats } from "@/components/documents/doc-stats";
import { DocTable } from "@/components/documents/doc-table";
import { useDocuments } from "@/hooks/use-documents";

export default function DocumentsPage() {
  const { data: documents, isLoading } = useDocuments();

  return (
    <div className="h-full overflow-auto flex flex-col gap-6 p-6">
      <div>
        <h1 className="text-2xl font-semibold">Documents</h1>
        <p className="text-sm text-muted-foreground">
          Browse and manage your indexed documents.
        </p>
      </div>

      {isLoading ? (
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-24 rounded-xl" />
            ))}
          </div>
          <Skeleton className="h-8 w-64" />
          <Skeleton className="h-64 rounded-xl" />
        </div>
      ) : documents && documents.length > 0 ? (
        <>
          <DocStats documents={documents} />
          <DocTable documents={documents} />
        </>
      ) : (
        <div className="flex flex-1 flex-col items-center justify-center gap-3 text-muted-foreground">
          <FolderOpen className="size-12 opacity-30" />
          <p className="text-sm">No documents yet. Upload some files first.</p>
        </div>
      )}
    </div>
  );
}
