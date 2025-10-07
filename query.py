#!/usr/bin/env python3
"""
Interactive CLI for querying the NoterLLM RAG system.
Provides a user-friendly interface for asking questions about Turkish notary regulations.
"""

import argparse
import sys
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.table import Table

# Import the RAG system
try:
    from llm_rag_setup import qa_chain, query_rag
    print("✅ RAG system loaded successfully!")
except Exception as e:
    print(f"❌ Error loading RAG system: {e}")
    print("Make sure you have run the setup and all dependencies are installed.")
    sys.exit(1)


def interactive_mode():
    """Run in interactive mode with continuous Q&A"""
    console = Console()
    
    console.print(Panel.fit(
        "[bold cyan]NoterLLM - Türk Noter Hukuku Soru-Cevap Sistemi[/bold cyan]\n"
        "Türkiye Noterler Birliği genelgelerine dayalı AI asistan\n\n"
        "[dim]Çıkmak için 'exit', 'quit' veya 'q' yazın[/dim]",
        border_style="cyan"
    ))
    print()
    
    query_count = 0
    
    while True:
        try:
            # Get user input
            question = Prompt.ask("\n[bold green]Sorunuz[/bold green]")
            
            # Check for exit commands
            if question.lower() in ['exit', 'quit', 'q', 'çıkış']:
                console.print("\n[yellow]Görüşmek üzere! 👋[/yellow]")
                break
            
            if not question.strip():
                console.print("[red]Lütfen bir soru girin.[/red]")
                continue
            
            # Query the RAG system
            query_count += 1
            print()
            result = query_rag(question)
            
            # Show query count
            console.print(f"\n[dim]Toplam soru sayısı: {query_count}[/dim]")
            
        except KeyboardInterrupt:
            console.print("\n\n[yellow]Program sonlandırılıyor...[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Hata:[/bold red] {e}")
            continue


def single_query_mode(question: str):
    """Run a single query and exit"""
    console = Console()
    
    console.print(Panel.fit(
        "[bold cyan]NoterLLM - Tek Soru Modu[/bold cyan]",
        border_style="cyan"
    ))
    print()
    
    result = query_rag(question)
    return result


def batch_mode(questions_file: str):
    """Process multiple questions from a file"""
    console = Console()
    
    console.print(Panel.fit(
        "[bold cyan]NoterLLM - Toplu Soru Modu[/bold cyan]",
        border_style="cyan"
    ))
    print()
    
    try:
        with open(questions_file, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f if line.strip()]
        
        console.print(f"📋 {len(questions)} soru yüklendi\n")
        
        results = []
        for i, question in enumerate(questions, 1):
            console.print(f"\n[bold blue]Soru {i}/{len(questions)}[/bold blue]")
            result = query_rag(question)
            results.append(result)
            console.print("\n" + "="*80)
        
        console.print(f"\n✅ {len(results)} soru işlendi")
        return results
        
    except FileNotFoundError:
        console.print(f"[red]❌ Dosya bulunamadı: {questions_file}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]❌ Hata: {e}[/red]")
        sys.exit(1)


def show_stats():
    """Show system statistics"""
    console = Console()
    
    # Import to get document count
    from llm_rag_setup import documents
    
    table = Table(title="NoterLLM Sistem İstatistikleri", show_header=True, header_style="bold cyan")
    table.add_column("Özellik", style="cyan")
    table.add_column("Değer", style="green")
    
    table.add_row("Toplam Chunk Sayısı", str(len(documents)))
    table.add_row("Embedding Modeli", "intfloat/multilingual-e5-base")
    table.add_row("LLM Modeli", "Gemini 1.5 Flash")
    table.add_row("Retrieval Stratejisi", "Hybrid (FAISS + BM25)")
    table.add_row("Top-K Retrieval", "5")
    
    console.print(table)


def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="NoterLLM - Türk Noter Hukuku Soru-Cevap Sistemi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  # İnteraktif mod (varsayılan)
  python query.py
  
  # Tek soru
  python query.py -q "Araç satış işlemlerinde hangi belgeler gereklidir?"
  
  # Toplu sorular (dosyadan)
  python query.py -b questions.txt
  
  # Sistem istatistikleri
  python query.py --stats
        """
    )
    
    parser.add_argument(
        '-q', '--query',
        type=str,
        help='Tek bir soru sor ve çık'
    )
    
    parser.add_argument(
        '-b', '--batch',
        type=str,
        metavar='FILE',
        help='Dosyadan toplu soru işle (her satırda bir soru)'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Sistem istatistiklerini göster'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='İnteraktif mod (varsayılan)'
    )
    
    args = parser.parse_args()
    
    # Determine mode
    if args.stats:
        show_stats()
    elif args.query:
        single_query_mode(args.query)
    elif args.batch:
        batch_mode(args.batch)
    else:
        # Default to interactive mode
        interactive_mode()


if __name__ == "__main__":
    main()

