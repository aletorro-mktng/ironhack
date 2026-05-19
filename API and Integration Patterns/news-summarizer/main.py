"""Main application entry point."""
import sys
from summarizer import NewsSummarizer, AsyncNewsSummarizer
import asyncio
 
def main():
    """Run the news summarizer."""
    print("="*80)
    print("NEWS SUMMARIZER - Multi-Provider Edition")
    print("="*80)
    
    # Get user input
    category = input("\nEnter news category (technology/business/health/general): ").strip() or "technology"
    num_articles = input("How many articles to process? (1-10): ").strip()
    
    try:
        num_articles = int(num_articles)
        num_articles = max(1, min(10, num_articles))  # Clamp between 1 and 10
    except:
        num_articles = 3
    
    use_async = input("Use async processing? (y/n): ").strip().lower() == 'y'
    
    print(f"\nFetching {num_articles} articles from category: {category}")
    
    try:
        if use_async:
            # Use async version
            summarizer = AsyncNewsSummarizer()
            articles = summarizer.news_api.fetch_top_headlines(
                category=category,
                max_articles=num_articles
            )
            
            if articles:
                print(f"\nProcessing {len(articles)} articles concurrently...")
                results = asyncio.run(
                    summarizer.process_articles_async(articles, max_concurrent=3)
                )
                summarizer.generate_report(results)
        
        else:
            # Use synchronous version
            summarizer = NewsSummarizer()
            articles = summarizer.news_api.fetch_top_headlines(
                category=category,
                max_articles=num_articles
            )
            
            if articles:
                print(f"\nProcessing {len(articles)} articles...")
                results = summarizer.process_articles(articles)
                summarizer.generate_report(results)
        
        print("\n✓ Processing complete!")
    
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
 
if __name__ == "__main__":
    main()
