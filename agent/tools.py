import arxiv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool


@tool
def search_arxiv(query: str) -> str:
    """
    Search arXiv for academic papers related to a query.
    Use this when the user asks about research papers, ML concepts,
    or wants to find academic references.
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = list(client.results(search))

        if not results:
            return "No papers found for this query."

        output = []
        for paper in results:
            output.append(
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(a.name for a in paper.authors[:3])}\n"
                f"Summary: {paper.summary[:300]}...\n"
                f"URL: {paper.entry_id}\n"
            )
        return "\n---\n".join(output)

    except Exception as e:
        return f"arXiv search failed: {str(e)}"


@tool
def search_web(query: str) -> str:
    """
    Search the web for current information, news, or general knowledge.
    Use this when the user asks about something not in academic papers,
    or wants recent information.
    """
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        return result if result else "No results found."
    except Exception as e:
        return f"Web search failed: {str(e)}"


# Export tools as a list
tools = [search_arxiv, search_web]
