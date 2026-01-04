import { useInfiniteQuery, useQuery } from '@tanstack/react-query'
import { Link, createFileRoute } from '@tanstack/react-router'
import { Layers } from 'lucide-react'
import { useMemo, useState } from 'react'
import type { BookmarkWithFeature } from '@/api/bookmarks'
import { fetchBookmarks } from '@/api/bookmarks'
import { featureQueryOptions } from '@/hooks/useFeatures'
import { Card } from '@/components/ui/card'
import { Spinner } from '@/components/ui/spinner'
import { FeatureCard } from '@/components/feature/feature-card'
import { FeatureBookmarkButton } from '@/components/feature/bookmark-button'
import { FeatureActivationSample } from '@/components/feature/sample'
import { cn } from '@/lib/utils'

export const Route = createFileRoute('/bookmarks/')({
  component: BookmarksPage,
  staticData: {
    fullScreen: true,
  },
})

function BookmarkFeatureItem({
  bookmark,
  isSelected,
  onClick,
}: {
  bookmark: BookmarkWithFeature
  isSelected: boolean
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'w-full text-left transition-all duration-150 cursor-pointer relative',
        'hover:bg-slate-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-sky-700/50',
        'border-b border-slate-3',
        isSelected &&
          'bg-slate-100 hover:bg-slate-100 inset-ring-2 inset-ring-sky-600 z-10',
      )}
    >
      <div className="flex flex-col gap-2 p-2">
        <div className="flex items-center justify-between gap-1">
          {!bookmark.feature?.interpretation && (
            <div className="font-token font-medium text-sm rounded-md w-fit text-neutral-500">
              N/A
            </div>
          )}
          {bookmark.feature?.interpretation && (
            <div className="font-token font-medium text-sm rounded-md w-fit">
              {bookmark.feature.interpretation.text}
            </div>
          )}
          <div className="flex items-center text-slate-500 text-xs">
            <span>#{bookmark.featureIndex}@</span>
            <span className="text-slate-400">{bookmark.saeName}</span>
          </div>
        </div>
        {bookmark.feature?.actTimes > 0 &&
        bookmark.feature?.samples.length > 0 ? (
          <FeatureActivationSample
            sample={bookmark.feature.samples[0]}
            maxFeatureAct={bookmark.feature.maxFeatureAct}
            visibleRange={5}
            showHighestActivatingToken={false}
            showHoverCard={false}
            className="px-2 pointer-events-none"
            sampleTextClassName="text-xs line-clamp-1"
          />
        ) : (
          <div className="flex flex-wrap whitespace-pre-wrap text-xs leading-relaxed font-mono text-slate-400 px-2 overflow-hidden text-ellipsis line-clamp-1">
            No activation
          </div>
        )}
      </div>
    </button>
  )
}

function BookmarksPage() {
  const [selectedBookmark, setSelectedBookmark] = useState<{
    saeName: string
    featureIndex: number
  } | null>(null)

  const {
    data: pagedData,
    isLoading: isLoadingBookmarks,
    hasNextPage,
    fetchNextPage,
    isFetchingNextPage,
  } = useInfiniteQuery({
    queryKey: ['bookmarks'],
    queryFn: async ({ pageParam = 0 }) => {
      const result = await fetchBookmarks({
        data: { limit: 25, skip: pageParam },
      })
      return result
    },
    getNextPageParam: (lastPage, allPages) => {
      const totalLoaded = allPages.reduce(
        (sum, page) => sum + page.bookmarks.length,
        0,
      )
      return totalLoaded < lastPage.totalCount ? totalLoaded : undefined
    },
    initialPageParam: 0,
  })

  const bookmarks = useMemo(
    () => pagedData?.pages.flatMap((page) => page.bookmarks) ?? [],
    [pagedData],
  )

  const totalCount = pagedData?.pages[0]?.totalCount ?? 0

  const { data: fullFeatureData, isLoading: isLoadingFeature } = useQuery({
    ...featureQueryOptions({
      dictionary: selectedBookmark?.saeName ?? '',
      featureIndex: selectedBookmark?.featureIndex ?? 0,
    }),
    enabled: !!selectedBookmark,
  })

  const handleSelectFeature = (bookmark: BookmarkWithFeature) => {
    setSelectedBookmark({
      saeName: bookmark.saeName,
      featureIndex: bookmark.featureIndex,
    })
  }

  return (
    <div className="h-full overflow-y-auto bg-linear-to-br from-slate-50 to-slate-100">
      <div className="w-[1400px] mx-auto p-8">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-slate-800 mb-2">Bookmarks</h1>
          <p className="text-slate-500">
            View and manage your bookmarked features
          </p>
        </div>

        <Card className="flex h-[750px] overflow-hidden">
          {/* Left Panel: Bookmarks List */}
          <div className="min-w-[350px] basis-[350px] shrink-0 flex flex-col border-r border-slate-300 bg-white">
            <div className="w-full h-[50px] uppercase px-4 flex items-center justify-between border-b border-b-slate-300 shrink-0 font-semibold tracking-tight text-sm text-slate-700 cursor-default">
              <span>Bookmarks ({totalCount})</span>
            </div>

            <div className="overflow-y-auto grow">
              {isLoadingBookmarks && (
                <div className="flex flex-col items-center justify-center h-full">
                  <Spinner isAnimating={true} />
                </div>
              )}

              {!isLoadingBookmarks && bookmarks.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full p-6 text-center">
                  <Layers className="w-8 h-8 mb-2 opacity-50 text-slate-400" />
                  <p className="text-slate-500 text-sm font-medium">
                    No bookmarks yet
                  </p>
                  <p className="text-slate-400 text-xs mt-1">
                    Bookmark features to see them here
                  </p>
                </div>
              )}

              {!isLoadingBookmarks && bookmarks.length > 0 && (
                <>
                  {bookmarks.map((bookmark) => (
                    <BookmarkFeatureItem
                      key={`${bookmark.saeName}-${bookmark.featureIndex}`}
                      bookmark={bookmark}
                      isSelected={
                        selectedBookmark?.saeName === bookmark.saeName &&
                        selectedBookmark?.featureIndex === bookmark.featureIndex
                      }
                      onClick={() => handleSelectFeature(bookmark)}
                    />
                  ))}
                  {hasNextPage && (
                    <div className="h-8 w-full flex items-center justify-center p-1">
                      <button
                        type="button"
                        onClick={() => fetchNextPage()}
                        disabled={isFetchingNextPage}
                        className="text-xs text-slate-500 hover:text-slate-700"
                      >
                        {isFetchingNextPage ? (
                          <Spinner
                            isAnimating={true}
                            className="text-slate-400"
                          />
                        ) : (
                          'Load more...'
                        )}
                      </button>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Right Panel: Feature Detail */}
          <div className="flex flex-col grow min-w-0 basis-0">
            {selectedBookmark && fullFeatureData && (
              <>
                <div className="relative w-full h-[50px] uppercase px-4 flex items-center justify-center text-sm gap-1 border-b border-b-slate-300 shrink-0 font-semibold tracking-tight text-slate-700 cursor-default">
                  <div className="absolute left-4 top-1/2 -translate-y-1/2">
                    <FeatureBookmarkButton feature={fullFeatureData} />
                  </div>
                  Feature{' '}
                  <Link
                    to={'/dictionaries/$dictionaryName/features/$featureIndex'}
                    params={{
                      dictionaryName: selectedBookmark.saeName,
                      featureIndex: selectedBookmark.featureIndex.toString(),
                    }}
                    className="text-sky-600 hover:text-sky-700"
                  >
                    #{selectedBookmark.featureIndex}
                  </Link>{' '}
                  from{' '}
                  <Link
                    to={'/dictionaries/$dictionaryName'}
                    params={{ dictionaryName: selectedBookmark.saeName }}
                    className="text-sky-600 hover:text-sky-700"
                  >
                    {selectedBookmark.saeName.replace('_', '-')}
                  </Link>
                  <Link
                    to={'/dictionaries/$dictionaryName/features/$featureIndex'}
                    params={{
                      dictionaryName: selectedBookmark.saeName,
                      featureIndex: selectedBookmark.featureIndex.toString(),
                    }}
                    className="absolute right-4 top-1/2 -translate-y-1/2 text-sky-600 hover:text-sky-700 text-xs"
                  >
                    Show Detail
                  </Link>
                </div>
                <FeatureCard
                  feature={fullFeatureData}
                  className="overflow-y-auto grow [scrollbar-gutter:stable] rounded-none border-none"
                />
              </>
            )}

            {selectedBookmark && isLoadingFeature && (
              <div className="flex flex-col items-center justify-center h-full">
                <Spinner isAnimating={true} />
              </div>
            )}

            {!selectedBookmark && (
              <div className="flex flex-col items-center justify-center h-full self-center">
                <p className="text-slate-500 text-sm font-medium">
                  Select a bookmark to view details
                </p>
                <p className="text-slate-400 text-xs mt-1">
                  Click on any bookmark from the list
                </p>
              </div>
            )}
          </div>
        </Card>
      </div>
    </div>
  )
}
