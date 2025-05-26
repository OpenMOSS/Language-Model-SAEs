import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { useState, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import camelcaseKeys from "camelcase-keys";

interface Bookmark {
  saeName: string;
  saeSeries: string;
  featureIndex: number;
  createdAt: string;
  tags: string[];
  notes?: string;
}

interface BookmarksResponse {
  bookmarks: Bookmark[];
  totalCount: number;
}

const BookmarksPage = () => {
  const [bookmarks, setBookmarks] = useState<Bookmark[]>([]);
  const [totalCount, setTotalCount] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState<number>(0);
  const [limit] = useState<number>(20);

  const fetchBookmarks = useCallback(async (pageNumber: number = 0) => {
    try {
      setLoading(true);
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/bookmarks?limit=${limit}&skip=${pageNumber * limit}`
      );

      if (!response.ok) {
        throw new Error("Failed to fetch bookmarks");
      }

      const rawData = await response.json();
      const data: BookmarksResponse = camelcaseKeys(rawData, { deep: true });
      setBookmarks(data.bookmarks);
      setTotalCount(data.totalCount);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  }, [limit]);

  const removeBookmark = async (saeName: string, featureIndex: number) => {
    try {
      const response = await fetch(
        `${import.meta.env.VITE_BACKEND_URL}/dictionaries/${saeName}/features/${featureIndex}/bookmark`,
        {
          method: "DELETE",
        }
      );

      if (response.ok) {
        // Refresh the bookmarks list
        await fetchBookmarks(page);
      } else {
        throw new Error("Failed to remove bookmark");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to remove bookmark");
    }
  };

  useEffect(() => {
    fetchBookmarks(page);
  }, [page, fetchBookmarks]);

  const totalPages = Math.ceil(totalCount / limit);

  if (loading && bookmarks.length === 0) {
    return (
      <div className="container mx-auto p-8">
        <div className="text-center">Loading bookmarks...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="container mx-auto p-8">
        <div className="text-center text-red-500">Error: {error}</div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-8">
      <Card>
        <CardHeader>
          <CardTitle className="flex justify-between items-center">
            <span>Bookmarked Features</span>
            <span className="text-sm text-muted-foreground">
              {totalCount} bookmark{totalCount !== 1 ? "s" : ""}
            </span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          {bookmarks.length === 0 ? (
            <div className="text-center py-8">
              <p className="text-muted-foreground">No bookmarks found.</p>
              <p className="text-sm text-muted-foreground mt-2">
                Start exploring features and bookmark interesting ones!
              </p>
            </div>
          ) : (
            <>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Feature</TableHead>
                    <TableHead>Dictionary</TableHead>
                    <TableHead>Series</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead>Tags</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {bookmarks.map((bookmark) => (
                    <TableRow key={`${bookmark.saeName}-${bookmark.featureIndex}`}>
                      <TableCell>
                        <Link
                          to={`/features?dictionary=${bookmark.saeName}&featureIndex=${bookmark.featureIndex}`}
                          className="text-blue-600 hover:underline font-medium"
                        >
                          #{bookmark.featureIndex}
                        </Link>
                      </TableCell>
                      <TableCell>{bookmark.saeName}</TableCell>
                      <TableCell>{bookmark.saeSeries}</TableCell>
                      <TableCell>
                        {new Date(bookmark.createdAt).toLocaleDateString()}
                      </TableCell>
                      <TableCell>
                        {bookmark.tags.length > 0 ? (
                          <div className="flex gap-1">
                            {bookmark.tags.map((tag, index) => (
                              <span
                                key={index}
                                className="inline-block bg-gray-100 text-gray-800 text-xs px-2 py-1 rounded"
                              >
                                {tag}
                              </span>
                            ))}
                          </div>
                        ) : (
                          <span className="text-muted-foreground">-</span>
                        )}
                      </TableCell>
                      <TableCell>
                        <Button
                          variant="destructive"
                          size="sm"
                          onClick={() => removeBookmark(bookmark.saeName, bookmark.featureIndex)}
                        >
                          Remove
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>

              {totalPages > 1 && (
                <div className="flex justify-center gap-2 mt-4">
                  <Button
                    variant="outline"
                    onClick={() => setPage(page - 1)}
                    disabled={page === 0}
                  >
                    Previous
                  </Button>
                  <span className="flex items-center px-4">
                    Page {page + 1} of {totalPages}
                  </span>
                  <Button
                    variant="outline"
                    onClick={() => setPage(page + 1)}
                    disabled={page >= totalPages - 1}
                  >
                    Next
                  </Button>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
};

export default BookmarksPage; 