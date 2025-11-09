import os
import json
import time
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup
import google.generativeai as genai
from duckduckgo_search import DDGS
import logging
from urllib.parse import urlparse
import hashlib
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Source:
    """Represents a research source"""
    url: str
    title: str
    content: str
    summary: str
    relevance_score: float
    extracted_date: str

@dataclass
class ResearchReport:
    """Represents the final research report"""
    topic: str
    executive_summary: str
    key_findings: List[str]
    detailed_analysis: str
    sources: List[Source]
    recommendations: List[str]
    generated_date: str

class GeminiAI:
    """Wrapper class for Google Gemini AI interactions"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.enabled = True
            logger.info("Gemini AI initialized successfully")
        else:
            self.model = None
            self.enabled = False
            logger.warning("No Gemini API key found. Using fallback methods.")
    
    def generate_content(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate content using Gemini AI"""
        if not self.enabled:
            return ""
        
        try:
            # Configure generation settings
            generation_config = genai.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=0.7,
                top_p=0.8,
                top_k=40
            )
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini AI generation error: {e}")
            return ""
    
    def summarize(self, content: str, max_sentences: int = 3) -> str:
        """Summarize content using Gemini"""
        if not self.enabled:
            return self._basic_summarize(content, max_sentences)
        
        prompt = f"""Summarize the following content in {max_sentences} clear and concise sentences:

Content: {content[:3000]}

Summary:"""
        
        summary = self.generate_content(prompt, max_tokens=200)
        return summary if summary else self._basic_summarize(content, max_sentences)
    
    def extract_key_points(self, content: str, num_points: int = 5) -> List[str]:
        """Extract key points from content"""
        if not self.enabled:
            return []
        
        prompt = f"""Extract {num_points} key points from the following content. 
Return each point as a single, complete sentence.

Content: {content[:3000]}

Key Points:"""
        
        response = self.generate_content(prompt, max_tokens=300)
        if response:
            # Parse bullet points or numbered lists
            points = re.findall(r'[\•\-\*\d+\.]\s*(.+)', response)
            if not points:
                # If no bullets found, split by newlines
                points = [p.strip() for p in response.split('\n') if p.strip()]
            return points[:num_points]
        return []
    
    def analyze_content(self, topic: str, content: str) -> str:
        """Perform detailed analysis of content"""
        if not self.enabled:
            return f"Analysis of {topic} based on available sources."
        
        prompt = f"""Provide a detailed analysis of the following information about "{topic}".
Include insights, patterns, and important observations.

Information: {content[:4000]}

Detailed Analysis:"""
        
        analysis = self.generate_content(prompt, max_tokens=500)
        return analysis if analysis else f"Analysis of {topic} based on available sources."
    
    def generate_recommendations(self, topic: str, findings: str) -> List[str]:
        """Generate recommendations based on research"""
        if not self.enabled:
            return self._default_recommendations(topic)
        
        prompt = f"""Based on research about "{topic}" and the following findings, 
provide 5 actionable recommendations:

Findings: {findings[:2000]}

Recommendations:"""
        
        response = self.generate_content(prompt, max_tokens=300)
        if response:
            recs = re.findall(r'[\•\-\*\d+\.]\s*(.+)', response)
            if not recs:
                recs = [r.strip() for r in response.split('\n') if r.strip()]
            return recs[:5]
        return self._default_recommendations(topic)
    
    def _basic_summarize(self, content: str, max_sentences: int) -> str:
        """Basic extractive summarization without AI"""
        if not content:
            return ""
        sentences = content.split('. ')[:max_sentences]
        return '. '.join(sentences) + '.' if sentences else ""
    
    def _default_recommendations(self, topic: str) -> List[str]:
        """Default recommendations when AI is not available"""
        return [
            f"Conduct further research on specific aspects of {topic}",
            "Verify findings through additional authoritative sources",
            "Consider practical applications of the research findings",
            "Analyze trends and patterns in the collected data",
            "Develop an action plan based on key findings"
        ]

class WebSearcher:
    """Handles web searching functionality"""
    
    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self.ddgs = DDGS()
    
    def search(self, query: str) -> List[Dict[str, str]]:
        """Search the web for relevant sources"""
        logger.info(f"Searching for: {query}")
        try:
            results = []
            search_results = self.ddgs.text(
                query, 
                region='wt-wt', 
                safesearch='moderate', 
                max_results=self.max_results
            )
            
            for result in search_results:
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('link', ''),
                    'snippet': result.get('body', '')
                })
            
            logger.info(f"Found {len(results)} search results")
            return results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def search_news(self, query: str) -> List[Dict[str, str]]:
        """Search for recent news articles"""
        logger.info(f"Searching news for: {query}")
        try:
            results = []
            news_results = self.ddgs.news(
                query,
                region='wt-wt',
                safesearch='moderate',
                max_results=5
            )
            
            for result in news_results:
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'snippet': result.get('body', ''),
                    'date': result.get('date', '')
                })
            
            return results
        except Exception as e:
            logger.error(f"News search error: {e}")
            return []

class ContentExtractor:
    """Extracts and processes content from web pages"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def extract_content(self, url: str) -> str:
        """Extract main content from a webpage"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Try to find main content areas
            main_content = soup.find(['main', 'article'])
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                # Fallback to all paragraphs
                paragraphs = soup.find_all(['p', 'div'])
                text = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            
            # Clean up text
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n+', ' ', text)
            
            # Limit content length
            max_chars = 5000
            if len(text) > max_chars:
                text = text[:max_chars] + "..."
            
            return text
        except Exception as e:
            logger.error(f"Content extraction error for {url}: {e}")
            return ""
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL is valid and accessible"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

class ContentAnalyzer:
    """Analyzes and summarizes content using Gemini AI"""
    
    def __init__(self, gemini_ai: GeminiAI):
        self.ai = gemini_ai
    
    def summarize_content(self, content: str, max_sentences: int = 3) -> str:
        """Summarize content using Gemini AI"""
        if not content:
            return ""
        
        return self.ai.summarize(content, max_sentences)
    
    def calculate_relevance(self, content: str, topic: str) -> float:
        """Calculate relevance score of content to topic"""
        if not content:
            return 0.0
        
        # Basic keyword matching
        topic_words = set(topic.lower().split())
        content_lower = content.lower()
        content_words = set(content_lower.split())
        
        if not topic_words:
            return 0.0
        
        # Calculate word overlap
        common_words = topic_words.intersection(content_words)
        relevance = len(common_words) / len(topic_words)
        
        # Bonus for exact phrase match
        if topic.lower() in content_lower:
            relevance = min(1.0, relevance + 0.3)
        
        # Additional bonus for multiple occurrences
        topic_count = content_lower.count(topic.lower())
        if topic_count > 1:
            relevance = min(1.0, relevance + (topic_count * 0.05))
        
        return min(1.0, relevance)
    
    def extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities from content"""
        entities = {
            'dates': re.findall(r'\b\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', content),
            'urls': re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content),
            'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content),
        }
        return entities

class ReportGenerator:
    """Generates the final research report using Gemini AI"""
    
    def __init__(self, analyzer: ContentAnalyzer, gemini_ai: GeminiAI):
        self.analyzer = analyzer
        self.ai = gemini_ai
    
    def generate_report(self, topic: str, sources: List[Source]) -> ResearchReport:
        """Generate a comprehensive research report"""
        logger.info("Generating research report with Gemini AI...")
        
        # Sort sources by relevance
        sources.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(topic, sources)
        
        # Extract key findings
        key_findings = self._extract_key_findings(topic, sources)
        
        # Generate detailed analysis
        detailed_analysis = self._generate_detailed_analysis(topic, sources)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(topic, sources, key_findings)
        
        return ResearchReport(
            topic=topic,
            executive_summary=executive_summary,
            key_findings=key_findings,
            detailed_analysis=detailed_analysis,
            sources=sources[:7],  # Top 7 sources
            recommendations=recommendations,
            generated_date=datetime.now().isoformat()
        )
    
    def _generate_executive_summary(self, topic: str, sources: List[Source]) -> str:
        """Generate executive summary using Gemini"""
        if not sources:
            return f"No relevant information found for the topic: {topic}"
        
        # Combine top source summaries
        top_summaries = [s.summary for s in sources[:5] if s.summary]
        combined_info = " ".join(top_summaries)
        
        if self.ai.enabled:
            prompt = f"""Write a comprehensive executive summary about "{topic}" based on the following research findings.
The summary should be 3-4 sentences that capture the most important insights.

Research findings:
{combined_info[:2000]}

Executive Summary:"""
            
            summary = self.ai.generate_content(prompt, max_tokens=250)
            if summary:
                return summary
        
        # Fallback to basic summary
        return f"Research on {topic} reveals important insights based on {len(sources)} sources analyzed. " + \
               f"{top_summaries[0] if top_summaries else 'Limited information available.'}"
    
    def _extract_key_findings(self, topic: str, sources: List[Source]) -> List[str]:
        """Extract key findings using Gemini AI"""
        if not sources:
            return []
        
        # Combine relevant content
        relevant_content = " ".join([s.summary for s in sources[:5] if s.relevance_score > 0.3])
        
        if self.ai.enabled and relevant_content:
            findings = self.ai.extract_key_points(relevant_content, num_points=5)
            if findings:
                return findings
        
        # Fallback to extracting from summaries
        findings = []
        for source in sources[:5]:
            if source.summary and source.relevance_score > 0.3:
                first_sentence = source.summary.split('.')[0] + '.'
                if len(first_sentence) > 20:
                    findings.append(first_sentence)
        
        return findings[:5]
    
    def _generate_detailed_analysis(self, topic: str, sources: List[Source]) -> str:
        """Generate detailed analysis section using Gemini"""
        if not sources:
            return "No sources available for detailed analysis."
        
        # Prepare content for analysis
        high_relevance = [s for s in sources if s.relevance_score > 0.6]
        medium_relevance = [s for s in sources if 0.3 < s.relevance_score <= 0.6]
        
        analysis_content = []
        if high_relevance:
            analysis_content.append("Primary sources: " + " ".join([s.summary for s in high_relevance[:3]]))
        if medium_relevance:
            analysis_content.append("Supporting sources: " + " ".join([s.summary for s in medium_relevance[:2]]))
        
        combined_content = " ".join(analysis_content)
        
        if self.ai.enabled and combined_content:
            analysis = self.ai.analyze_content(topic, combined_content)
            if analysis:
                return f"## Detailed Analysis: {topic}\n\n{analysis}"
        
        # Fallback to structured analysis
        analysis_parts = [
            f"## Detailed Analysis: {topic}\n",
            f"\nBased on analysis of {len(sources)} sources:\n"
        ]
        
        if high_relevance:
            analysis_parts.append("\n### Primary Findings:\n")
            for source in high_relevance[:3]:
                analysis_parts.append(f"- {source.summary}\n")
        
        if medium_relevance:
            analysis_parts.append("\n### Supporting Information:\n")
            for source in medium_relevance[:2]:
                analysis_parts.append(f"- {source.summary}\n")
        
        return "".join(analysis_parts)
    
    def _generate_recommendations(self, topic: str, sources: List[Source], key_findings: List[str]) -> List[str]:
        """Generate recommendations using Gemini AI"""
        if not sources:
            return ["Conduct additional research with broader search parameters"]
        
        # Prepare findings summary
        findings_summary = " ".join(key_findings[:3]) if key_findings else "Limited findings available."
        
        if self.ai.enabled:
            recommendations = self.ai.generate_recommendations(topic, findings_summary)
            if recommendations:
                return recommendations
        
        # Fallback recommendations
        recommendations = [
            f"Further investigate specific aspects of {topic}",
            "Verify findings through peer-reviewed sources",
            "Consider practical applications of the research"
        ]
        
        if len(sources) < 3:
            recommendations.append("Expand search parameters for more comprehensive data")
        
        if any(s.relevance_score > 0.7 for s in sources):
            recommendations.append("Focus on high-relevance sources for implementation")
        
        return recommendations[:5]

class ResearchAssistant:
    """Main AI Agent that orchestrates the research process"""
    
    def __init__(self, gemini_api_key: str = None):
        self.gemini_ai = GeminiAI(gemini_api_key)
        self.web_searcher = WebSearcher(max_results=10)
        self.content_extractor = ContentExtractor()
        self.content_analyzer = ContentAnalyzer(self.gemini_ai)
        self.report_generator = ReportGenerator(self.content_analyzer, self.gemini_ai)
        self.cache = {}
        
    def research(self, topic: str, include_news: bool = True, deep_search: bool = True) -> ResearchReport:
        """Main method to conduct research on a topic"""
        logger.info(f"Starting research on topic: {topic}")
        logger.info(f"Using Gemini AI: {self.gemini_ai.enabled}")
        
        # Generate search queries
        search_queries = self._generate_search_queries(topic)
        
        # Collect sources
        all_sources = []
        
        # Web search
        for query in search_queries:
            search_results = self.web_searcher.search(query)
            
            for result in search_results:
                source = self._process_source(result, topic)
                if source and source.relevance_score > 0.2:
                    all_sources.append(source)
            
            time.sleep(1)  # Rate limiting
        
        # News search (if enabled)
        if include_news:
            news_results = self.web_searcher.search_news(topic)
            for result in news_results:
                source = self._process_source(result, topic)
                if source and source.relevance_score > 0.2:
                    all_sources.append(source)
        
        # Remove duplicates
        all_sources = self._remove_duplicate_sources(all_sources)
        
        if not all_sources:
            logger.warning(f"No sources found for topic: {topic}")
        
        # Generate report
        report = self.report_generator.generate_report(topic, all_sources)
        
        # Save report
        self._save_report(report)
        
        logger.info(f"Research completed. Report generated with {len(report.sources)} sources.")
        return report
    
    def _generate_search_queries(self, topic: str) -> List[str]:
        """Generate multiple search queries for comprehensive coverage"""
        base_queries = [
            topic,
            f"what is {topic}",
            f"{topic} explained",
            f"{topic} latest developments",
            f"{topic} research findings"
        ]
        
        # Use Gemini to generate additional queries if available
        if self.gemini_ai.enabled:
            prompt = f"""Generate 3 different search queries to research "{topic}". 
Return only the queries, one per line."""
            
            ai_queries = self.gemini_ai.generate_content(prompt, max_tokens=100)
            if ai_queries:
                additional = [q.strip() for q in ai_queries.split('\n') if q.strip()]
                base_queries.extend(additional[:3])
        
        return base_queries[:5]  # Return top 5 queries
    
    def _process_source(self, search_result: Dict[str, str], topic: str) -> Optional[Source]:
        """Process a search result into a Source object"""
        url = search_result.get('url', '')
        
        if not self.content_extractor.is_valid_url(url):
            return None
        
        # Check cache
        url_hash = hashlib.md5(url.encode()).hexdigest()
        if url_hash in self.cache:
            return self.cache[url_hash]
        
        try:
            # Extract content
            content = self.content_extractor.extract_content(url)
            if not content:
                content = search_result.get('snippet', '')
            
            # Summarize content
            summary = self.content_analyzer.summarize_content(content)
            if not summary:
                summary = search_result.get('snippet', '')[:200]
            
            # Calculate relevance
            relevance_score = self.content_analyzer.calculate_relevance(content, topic)
            
            source = Source(
                url=url,
                title=search_result.get('title', 'Untitled'),
                content=content[:1500],  # Store first 1500 chars
                summary=summary,
                relevance_score=relevance_score,
                extracted_date=datetime.now().isoformat()
            )
            
            # Cache the source
            self.cache[url_hash] = source
            
            return source
        except Exception as e:
            logger.error(f"Error processing source {url}: {e}")
            return None
    
    def _remove_duplicate_sources(self, sources: List[Source]) -> List[Source]:
        """Remove duplicate sources based on URL and similar content"""
        seen_urls = set()
        seen_titles = set()
        unique_sources = []
        
        for source in sources:
            # Check for URL duplicates
            if source.url in seen_urls:
                continue
                
            # Check for similar titles (fuzzy matching)
            title_words = set(source.title.lower().split())
            is_duplicate = False
            for seen_title in seen_titles:
                seen_words = set(seen_title.lower().split())
                if len(title_words.intersection(seen_words)) > len(title_words) * 0.7:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_urls.add(source.url)
                seen_titles.add(source.title)
                unique_sources.append(source)
        
        return unique_sources
    
    def _save_report(self, report: ResearchReport):
        """Save report to file in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"research_report_{timestamp}"
        
        # Save as Markdown
        md_filename = f"{base_filename}.md"
        with open(md_filename, 'w', encoding='utf-8') as f:
            f.write(f"# Research Report: {report.topic}\n\n")
            f.write(f"**Generated:** {report.generated_date}\n")
            f.write(f"**AI Model:** Google Gemini Pro\n\n")
            
            f.write(f"## Executive Summary\n\n{report.executive_summary}\n\n")
            
            f.write(f"## Key Findings\n\n")
            for i, finding in enumerate(report.key_findings, 1):
                f.write(f"{i}. {finding}\n")
            
            f.write(f"\n## Detailed Analysis\n\n{report.detailed_analysis}\n\n")
            
            f.write(f"## Sources\n\n")
            for i, source in enumerate(report.sources, 1):
                f.write(f"### {i}. {source.title}\n")
                f.write(f"- **URL:** {source.url}\n")
                f.write(f"- **Relevance Score:** {source.relevance_score:.2%}\n")
                f.write(f"- **Summary:** {source.summary}\n\n")
            
            f.write(f"## Recommendations\n\n")
            for i, rec in enumerate(report.recommendations, 1):
                f.write(f"{i}. {rec}\n")
        
        # Save as JSON
        json_filename = f"{base_filename}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            report_dict = {
                'topic': report.topic,
                'executive_summary': report.executive_summary,
                'key_findings': report.key_findings,
                'detailed_analysis': report.detailed_analysis,
                'sources': [
                    {
                        'title': s.title,
                        'url': s.url,
                        'summary': s.summary,
                        'relevance_score': s.relevance_score
                    } for s in report.sources
                ],
                'recommendations': report.recommendations,
                'generated_date': report.generated_date
            }
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Report saved to {md_filename} and {json_filename}")

class ResearchAssistantCLI:
    """Command-line interface for the Research Assistant"""
    
    def __init__(self, gemini_api_key: str = None):
        self.assistant = ResearchAssistant(gemini_api_key)
    
    def run(self):
        """Run the interactive CLI"""
        print("\n" + "="*70)
        print("🔍 AI Research Assistant powered by Google Gemini")
        print("="*70)
        print("\nWelcome! I can help you research any topic and generate")
        print("comprehensive reports with sources and AI-powered analysis.\n")
        
        if self.assistant.gemini_ai.enabled:
            print("✅ Gemini AI is enabled for enhanced analysis")
        else:
            print("⚠️  Running without Gemini AI (using basic analysis)")
            print("   Set GEMINI_API_KEY environment variable to enable AI features")
        
        while True:
            try:
                print("\n" + "-"*50)
                topic = input("\n📝 Enter research topic (or 'quit' to exit): ").strip()
                
                if topic.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using Research Assistant. Goodbye!")
                    break
                
                if not topic:
                    print("❌ Please enter a valid topic.")
                    continue
                
                # Ask for options
                include_news = input("Include recent news? (y/n, default=y): ").strip().lower()
                include_news = include_news != 'n'
                
                print(f"\n🔎 Researching '{topic}'...")
                print("⏳ This may take a few moments...\n")
                
                # Progress indicators
                print("📊 Progress:")
                print("  ✓ Generating search queries...")
                print("  ✓ Searching web sources...")
                if include_news:
                    print("  ✓ Searching news articles...")
                print("  ✓ Extracting content...")
                print("  ✓ Analyzing with Gemini AI..." if self.assistant.gemini_ai.enabled else "  ✓ Analyzing content...")
                print("  ✓ Generating report...\n")
                
                # Conduct research
                report = self.assistant.research(topic, include_news=include_news)
                
                # Display results
                self._display_report(report)
                
                # Ask for next action
                print("\n" + "-"*50)
                action = input("\n🔄 Press Enter for another topic, 'v' to view saved report, or 'quit' to exit: ").strip()
                
                if action.lower() == 'v':
                    self._show_saved_reports()
                elif action.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Thank you for using Research Assistant. Goodbye!")
                    break
                    
            except KeyboardInterrupt:
                print("\n\n⚠️  Research interrupted. Exiting...")
                break
            except Exception as e:
                print(f"\n❌ An error occurred: {e}")
                print("Please try again with a different topic.")
    
    def _display_report(self, report: ResearchReport):
        """Display the research report in the console"""
        print("\n" + "="*70)
        print(f"📊 RESEARCH REPORT: {report.topic.upper()}")
        print("="*70)
        
        print(f"\n📝 EXECUTIVE SUMMARY\n{'-'*50}")
        print(report.executive_summary)
        
        if report.key_findings:
            print(f"\n🎯 KEY FINDINGS\n{'-'*50}")
            for i, finding in enumerate(report.key_findings, 1):
                print(f"{i}. {finding}")
        
        print(f"\n📚 TOP SOURCES (Relevance Score)\n{'-'*50}")
        for i, source in enumerate(report.sources[:5], 1):
            print(f"{i}. {source.title}")
            print(f"   📊 Relevance: {source.relevance_score:.0%}")
            print(f"   🔗 {source.url[:60]}...")
        
        if report.recommendations:
            print(f"\n💡 RECOMMENDATIONS\n{'-'*50}")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"{i}. {rec}")
        
        print(f"\n✅ Report saved to files (Markdown & JSON)")
        print("="*70)
    
    def _show_saved_reports(self):
        """Show list of saved reports"""
        import glob
        reports = glob.glob("research_report_*.md")
        if reports:
            print("\n📁 Saved Reports:")
            for report in sorted(reports)[-5:]:  # Show last 5 reports
                print(f"  - {report}")
        else:
            print("\n❌ No saved reports found.")

# Example usage and testing
if __name__ == "__main__":
    # Set your Gemini API key here or as environment variable
    # You can get your API key from: https://makersuite.google.com/app/apikey
    
    # Option 1: Set directly (not recommended for production)
    # gemini_api_key = "AIzaSyDCOoJAMaaZvuABLFb8oLCz4fShbuhHdJo"
    
    # Option 2: Use environment variable (recommended)
    # export GEMINI_API_KEY="your-api-key" (in terminal)
    # or set in .env file
    
    gemini_api_key = None  # Will look for GEMINI_API_KEY env variable
    
    # Run interactive CLI
    cli = ResearchAssistantCLI(gemini_api_key)
    cli.run()
    
    # Or use programmatically:
    """
    assistant = ResearchAssistant(gemini_api_key)
    report = assistant.research(
        topic="Artificial Intelligence in Healthcare",
        include_news=True,
        deep_search=True
    )
    print(f"Research complete! Found {len(report.sources)} relevant sources.")
    print(f"Executive Summary: {report.executive_summary}")
    """