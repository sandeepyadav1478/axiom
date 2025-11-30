"""
Start All Data Streams for Dashboard
Populates real-time dashboard with live data
"""

import asyncio
import logging
import os
import sys

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from axiom.streaming.integrations import IntegratedStreamingPlatform

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Start all data streams."""
    
    # Symbols to monitor
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
    
    logger.info("="*60)
    logger.info("STARTING INTEGRATED DATA STREAMS")
    logger.info("="*60)
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Dashboard: http://localhost:8001/")
    logger.info("="*60)
    
    try:
        # Initialize platform
        platform = IntegratedStreamingPlatform()
        
        # Start monitoring
        await platform.start_full_monitoring(symbols)
        
        logger.info("\n✅ All data streams started!")
        logger.info("   - Price updates every 5 seconds")
        logger.info("   - Open http://localhost:8001/ to see live data")
        logger.info("   - Press Ctrl+C to stop\n")
        
        # Keep running
        while True:
            await asyncio.sleep(60)
            logger.info(f"📊 Streaming active for {len(symbols)} symbols...")
            
    except KeyboardInterrupt:
        logger.info("\n\nStopping data streams...")
        platform.stop_monitoring()
        logger.info("✅ Stopped cleanly")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == '__main__':
    asyncio.run(main())