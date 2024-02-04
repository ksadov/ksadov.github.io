--------------------------------------------------------------------------------
{-# LANGUAGE OverloadedStrings #-}
import           Data.Monoid (mappend)
import           Hakyll
import           Data.List                       (intersperse)


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
            posts <- chronological =<< loadAll pattern
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
            posts <- recentFirst =<< loadAll "posts/*"
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
    dateField "date" "%B %e, %Y" `mappend`
    defaultContext

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