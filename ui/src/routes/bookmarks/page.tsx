import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Combobox } from "@/components/ui/combobox";
import { useState, useEffect, useCallback, useMemo } from "react";
import { Link } from "react-router-dom";
import camelcaseKeys from "camelcase-keys";
import { useAsyncFn, useMount } from "react-use";
import { z } from "zod";

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
  const [selectedDictionary, setSelectedDictionary] = useState<string | null>(null);
  const [selectedAnalysis, setSelectedAnalysis] = useState<string | null>(null);

  const [dictionariesState, fetchDictionaries] = useAsyncFn(async () => {
    return await fetch(`${import.meta.env.VITE_BACKEND_URL}/dictionaries`)
      .then(async (res) => await res.json())
      .then((res) => z.array(z.string()).parse(res));
  });

  const [analysesState, fetchAnalyses] = useAsyncFn(async (dictionary: string) => {
    if (!dictionary) return [];

    return await fetch(`${import.meta.env.VITE_BACKEND_URL}/dictionaries/${dictionary}/analyses`)
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(await res.text());
        }
        return res;
      })
      .then(async (res) => await res.json())
      .then((res) => z.array(z.string()).parse(res));
  });

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

  useMount(async () => {
    await fetchDictionaries();
  });

  useEffect(() => {
    if (dictionariesState.value && dictionariesState.value.length > 0 && selectedDictionary === null) {
      setSelectedDictionary(dictionariesState.value[0]);
      fetchAnalyses(dictionariesState.value[0]).then((analyses) => {
        if (analyses.length > 0) {
          setSelectedAnalysis(analyses[0]);
        }
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dictionariesState.value]);

  useEffect(() => {
    if (selectedDictionary) {
      fetchAnalyses(selectedDictionary);
      setSelectedAnalysis(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedDictionary]);

  useEffect(() => {
    if (analysesState.value && analysesState.value.length > 0 && selectedAnalysis === null) {
      setSelectedAnalysis(analysesState.value[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysesState.value]);

  useEffect(() => {
    fetchBookmarks(page);
  }, [page, fetchBookmarks]);

  // Memoize dictionary options for Combobox
  const dictionaryOptions = useMemo(() => {
    if (!dictionariesState.value) return [];
    return dictionariesState.value.map((dict) => ({
      value: dict,
      label: dict,
    }));
  }, [dictionariesState.value]);

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
          <div className="mb-6 grid grid-cols-[auto_300px_auto_300px] justify-center items-center gap-4">
            <span className="font-bold justify-self-end">Select dictionary:</span>
            <Combobox
              disabled={dictionariesState.loading || loading}
              value={selectedDictionary || null}
              onChange={(value) => {
                setSelectedDictionary(value);
              }}
              options={dictionaryOptions}
              placeholder="选择字典..."
              commandPlaceholder="搜索字典..."
              emptyIndicator="未找到匹配的字典"
              className="w-full"
            />
            <span className="font-bold justify-self-end">Select analysis:</span>
            <Select
              disabled={analysesState.loading || !selectedDictionary || loading}
              value={selectedAnalysis || undefined}
              onValueChange={setSelectedAnalysis}
            >
              <SelectTrigger className="bg-white">
                <SelectValue placeholder="Select an analysis" />
              </SelectTrigger>
              <SelectContent>
                {analysesState.value?.map((analysis, i) => (
                  <SelectItem key={i} value={analysis}>
                    {analysis}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
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
                          to={`/features?dictionary=${bookmark.saeName}&featureIndex=${bookmark.featureIndex}${selectedAnalysis ? `&analysis=${selectedAnalysis}` : ""}`}
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