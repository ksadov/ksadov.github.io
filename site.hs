--------------------------------------------------------------------------------
{-# LANGUAGE OverloadedStrings, TupleSections #-}
import Data.Monoid (mappend)
import Hakyll
import Data.List (intersperse)
import Control.Applicative (empty)
import Control.Monad (void, (>>=), liftM)
import Data.Ord (comparing)
import Data.List(sortOn, sortBy)
import Data.Maybe(fromMaybe)
import Text.Read(readMaybe)
import Data.Foldable (toList)

--------------------------------------------------------------------------------
main :: IO ()
main = hakyllWith config $ do
    match "images/*" $ do
        route   idRoute
        compile copyFileCompiler

    match "css/*" $ do
        route   idRoute
        compile compressCssCompiler

    match (fromList ["about.rst", "contact.markdown"]) $ do
        route   $ setExtension "html"
        compile $ pandocCompiler
            >>= loadAndApplyTemplate "templates/default.html" defaultContext
            >>= relativizeUrls

    -- build up tags
    tags <- buildTags "posts/*" (fromCapture "tags/*.html")

    tagsRules tags $ \tag pattern -> do
        let title = "Posts tagged \"" ++ tag ++ "\""
        route idRoute
        compile $ do
            posts <- recentFirst =<< loadAll pattern
            let ctx = constField "title" title
                      `mappend` listField "posts" (postCtxWithTags tags) (return posts)
                      `mappend` defaultContext

            makeItem ""
                >>= loadAndApplyTemplate "templates/tag.html" ctx
                >>= loadAndApplyTemplate "templates/default.html" ctx
                >>= relativizeUrls

    -- build up series
    series <- buildSeries "posts/*" (fromCapture "series/*.html")

    tagsRules series $ \sr pattern -> do
        let title = "Posts in the series \"" ++ sr ++ "\""
        route idRoute
        compile $ do
            posts <- bySeriesIdx =<< loadAll pattern
            let ctx = constField "title" title
                      `mappend` listField "posts" (postCtxWithSeries series) (return posts)
                      `mappend` defaultContext

            makeItem ""
                >>= loadAndApplyTemplate "templates/series.html" ctx
                >>= loadAndApplyTemplate "templates/default.html" ctx
                >>= relativizeUrls

    match "posts/*" $ do
        route $ setExtension "html"
        compile $ pandocCompiler
            >>= loadAndApplyTemplate "templates/post.html"    (postCtxWithMeta tags series)
            >>= loadAndApplyTemplate "templates/default.html" (postCtxWithMeta tags series)
            >>= relativizeUrls

    create ["archive.html"] $ do
        route idRoute
        compile $ do
            posts <- recentFirst =<< loadAll "posts/*"
            let archiveCtx =
                    listField "posts" postCtx (return posts) `mappend`
                    constField "title" "Archives"            `mappend`
                    defaultContext

            makeItem ""
                >>= loadAndApplyTemplate "templates/archive.html" archiveCtx
                >>= loadAndApplyTemplate "templates/default.html" archiveCtx
                >>= relativizeUrls


    match "index.html" $ do
        route idRoute
        compile $ do
            posts <- fmap (take 5) . recentFirst =<< loadAll "posts/*"
            let indexCtx =
                    listField "posts" postCtx (return posts) `mappend`
                    defaultContext

            getResourceBody
                >>= applyAsTemplate indexCtx
                >>= loadAndApplyTemplate "templates/default.html" indexCtx
                >>= relativizeUrls

    match "templates/*" $ compile templateBodyCompiler


--------------------------------------------------------------------------------
postCtx :: Context String
postCtx =
    transformedMetadataField "series-idx" "series-idx" (id) `mappend`
    dateField "date" "%B %e, %Y" `mappend`
    constField "og-type" "article" `mappend`
    defaultContext

-- | Creates a new field based on the item's metadata. If the metadata
-- field is not present then no field will actually be created.
-- Otherwise, the value will be passed to the given function and the
-- result of that function will be used as the field's value.
transformedMetadataField :: String -> String -> (String -> String) -> Context a
transformedMetadataField newKey originalKey f = Context $ \k _ i -> do
    if k == newKey
       then do
           value <- getMetadataField (itemIdentifier i) originalKey
           maybe empty (return . StringField . f) value
       else empty

config :: Configuration
config = defaultConfiguration
    { destinationDirectory = "docs"
    , previewPort          = 8009
    }

postCtxWithTags :: Tags -> Context String
postCtxWithTags tags = tagsField "tags" tags `mappend` postCtx

getSeries :: MonadMetadata m => Identifier -> m [String]
getSeries = getTagsByField "series"

buildSeries :: Pattern -> (String -> Identifier) -> Rules Tags
buildSeries pattern makeId =
    buildTagsWith getSeries pattern makeId

seriesField :: String     -- ^ Destination key
          -> Tags       -- ^ Tags
          -> Context a  -- ^ Context
seriesField = tagsFieldWith getSeries simpleRenderLink (mconcat . intersperse ", ")

postCtxWithSeries :: Tags -> Context String
postCtxWithSeries tags = tagsField "series" tags `mappend` postCtx

postCtxWithMeta :: Tags -> Tags -> Context String
postCtxWithMeta tags series =
    tagsField "tags" tags `mappend`
    seriesField "series" series `mappend`
    postCtx

-- this parses the series-idx out of an item
seriesIdx :: MonadMetadata m => Item a -> m Int
seriesIdx i = do
    mStr <- getMetadataField (itemIdentifier i) "series-idx"
    return $ (fromMaybe 0 $ mStr >>= readMaybe)

bySeriesIdx :: MonadMetadata m => [Item a] -> m [Item a]
bySeriesIdx = sortByM seriesIdx
  where
    sortByM :: (Monad m, Ord k) => (a -> m k) -> [a] -> m [a]
    sortByM f xs = liftM (map fst . sortBy (comparing snd)) $
                   mapM (\x -> liftM (x,) (f x)) xs
