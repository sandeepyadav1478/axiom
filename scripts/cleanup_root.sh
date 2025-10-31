#!/bin/bash
# Professional Root Directory Cleanup

echo "Starting root directory cleanup..."

# Create necessary directories
mkdir -p docs/sessions docs/archive docs/mcp docs/platform docs/status scripts/mcp_testing

# Move SESSION files
for file in SESSION_*.md THREAD_*.md; do
    [ -f "$file" ] && mv "$file" docs/sessions/ && echo "Moved $file"
done

# Move old summary/status files  
for file in *_COMPLETE*.md *_SUMMARY*.md *_STATUS*.md *_ACHIEVEMENT*.md *_MILESTONE*.md FINAL_*.md COMPLETE_*.md; do
    [ -f "$file" ] && [[ ! "$file" =~ "PROFESSIONAL_MCP" ]] && mv "$file" docs/archive/ && echo "Moved $file"
done

# Move MCP documentation
for file in MCP_*.md ALL_12_MCP*.md; do
    [ -f "$file" ] && mv "$file" docs/mcp/ && echo "Moved $file"
done

# Move platform docs
for file in PLATFORM_*.md PROFESSIONAL_AGENT*.md MASTER_INDEX.md ML_MODELS*.md RESEARCH_*.md; do
    [ -f "$file" ] && mv "$file" docs/platform/ && echo "Moved $file"
done

# Move status files
for file in CURRENT_*.md NEXT_*.md STATUS.txt; do
    [ -f "$file" ] && mv "$file" docs/status/ && echo "Moved $file"
done

# Move test scripts
for file in test_*.py test_*.sh validate_*.py verify_*.py fix_*.sh; do
    [ -f "$file" ] && mv "$file" scripts/mcp_testing/ && echo "Moved $file"
done

# Move misc docs
for file in *_GUIDE.md *_PLAN.md URGENT_*.md FIX_*.md WORK_*.md *_WORK*.md GENERATE_*.md; do
    [ -f "$file" ] && [[ ! "$file" =~ "CLEANUP" ]] && [[ ! "$file" =~ "README" ]] && mv "$file" docs/archive/ && echo "Moved $file"
done

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "Files remaining in root:"
ls -1 *.md *.txt *.sh 2>/dev/null | wc -l