# Git Workflow Guidelines - Strict Procedures

## ⚠️ CRITICAL: Lesson Learned from Nov 20, 2025

**What Went Wrong:**
After PR #38 merged to main, I continued using the same feature branch (`feature/enterprise-airflow-enhancements-20251120`) for new work. This caused the new commit to appear on both the feature branch AND main, violating Rule #5.

**Why It Happened:**
- The feature branch was already merged into main
- Local and remote branches were synced
- New commits on that branch also appeared on main
- **Root cause:** Reusing a merged feature branch

---

## ✅ CORRECT Workflow After PR Merge

### Step 1: After Your PR Gets Merged
```bash
# Your PR #38 was just merged to main on GitHub

# 1. Switch to main branch locally
git checkout main

# 2. Pull the merged changes from GitHub
git pull origin main

# 3. Verify you're on main with latest changes
git log --oneline -3
# Should show the merge commit
```

### Step 2: Create NEW Feature Branch for New Work
```bash
# 4. Create a NEW feature branch (don't reuse old one!)
git checkout -b feature/new-feature-name-20251120

# NEVER reuse a merged branch like this:
# git checkout feature/enterprise-airflow-enhancements-20251120  ❌ WRONG!

# 5. Verify you're on the new branch
git branch --show-current
# Should show: feature/new-feature-name-20251120
```

### Step 3: Make Your Changes
```bash
# 6. Make your code changes
# Edit files...

# 7. Stage changes
git add .

# 8. MANDATORY: Verify you're NOT on main
git branch --show-current
# If shows "main" → STOP! Create feature branch first!

# 9. Commit (only if NOT on main)
git commit -m "Your commit message"

# 10. Push to NEW feature branch
git push origin feature/new-feature-name-20251120
```

### Step 4: Create New PR
```bash
# 11. Create PR on GitHub
# From: feature/new-feature-name-20251120
# To: main

# 12. After PR merges, delete the feature branch
git branch -d feature/new-feature-name-20251120
git push origin --delete feature/new-feature-name-20251120
```

---

## 🚨 Common Mistakes to Avoid

### ❌ MISTAKE #1: Reusing Merged Feature Branches
```bash
# PR #38 merged to main
git checkout feature/enterprise-airflow-enhancements-20251120  # ❌ OLD merged branch!
# Make changes
git commit -m "New work"  # ❌ Goes to main too!
```

**Why it's bad:** Merged branches share history with main. New commits appear everywhere.

**Correct approach:**
```bash
git checkout main
git checkout -b feature/new-work-20251120  # ✅ Fresh branch from main
```

---

### ❌ MISTAKE #2: Committing on Main Branch
```bash
git branch --show-current  # Shows "main"
git commit -m "Quick fix"  # ❌ DIRECT TO MAIN!
git push origin main       # ❌ RULE #5 VIOLATION!
```

**Why it's bad:** Bypasses code review, breaks project rules, no PR history.

**Correct approach:**
```bash
git branch --show-current  # Shows "main"
# STOP! Create feature branch first:
git checkout -b feature/quick-fix-20251120  # ✅
git commit -m "Quick fix"  # ✅ On feature branch
git push origin feature/quick-fix-20251120  # ✅ To feature branch
```

---

### ❌ MISTAKE #3: Not Checking Branch Before Commit
```bash
# Assuming you're on feature branch without checking
git commit -m "Changes"  # Might be on main!
```

**Why it's bad:** Accidental commits to main.

**Correct approach:**
```bash
# ALWAYS check before committing
git branch --show-current  # ✅ Verify branch

# If on main:
echo "⚠️ STOP! On main branch!"
git checkout -b feature/my-work-20251120

# If on feature branch:
echo "✅ Safe to commit"
git commit -m "Changes"
```

---

## 📋 Pre-Commit Checklist (MANDATORY)

**Before EVERY commit, verify:**

```bash
# 1. Check current branch
git branch --show-current

# 2. Verify NOT on main
if [ "$(git branch --show-current)" == "main" ]; then
    echo "⛔ STOP! Cannot commit to main!"
    echo "Run: git checkout -b feature/your-feature-20251120"
    exit 1
fi

# 3. Check what will be committed
git status

# 4. Review changes
git diff

# 5. Only then commit
git commit -m "Your message"
```

---

## 🔧 Git Hooks for Automation

### Pre-Commit Hook (Prevents Main Branch Commits)

Create `.git/hooks/pre-commit`:

```bash
#!/bin/bash

# Prevent commits to main branch
BRANCH=$(git branch --show-current)

if [ "$BRANCH" == "main" ]; then
    echo "⛔ ERROR: Cannot commit directly to main branch!"
    echo ""
    echo "Please create a feature branch:"
    echo "  git checkout -b feature/your-feature-$(date +%Y%m%d)"
    echo ""
    echo "This enforces Rule #5: NEVER push directly to main"
    exit 1
fi

echo "✅ Branch check passed: $BRANCH"
exit 0
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

---

## 📊 Branch Lifecycle

```
1. Create Feature Branch
   ↓
2. Make Changes & Commits
   ↓
3. Push to Feature Branch
   ↓
4. Create Pull Request
   ↓
5. Code Review
   ↓
6. PR Merges to Main
   ↓
7. DELETE Feature Branch ← Important!
   ↓
8. Pull Latest Main
   ↓
9. Create NEW Feature Branch for next work
   (NEVER reuse the merged one)
```

---

## 💡 Best Practices

### 1. One Feature = One Branch
- Each new feature/fix gets its own branch
- Delete branch after PR merges
- Start fresh for next feature

### 2. Always Pull Main Before Creating Branch
```bash
git checkout main
git pull origin main  # Get latest
git checkout -b feature/new-work-20251120
```

### 3. Use Descriptive Branch Names
```bash
# ✅ Good
feature/enterprise-airflow-enhancements-20251120
feature/github-auto-merge-workflow-20251120
hotfix/fix-claude-api-key-20251120

# ❌ Bad
feature/test
feature/updates
fix
```

### 4. Keep Feature Branches Short-Lived
- Create → Commit → Push → PR → Merge → Delete
- Don't keep branches for weeks
- Reduces merge conflicts

---

## 🎯 Summary: The Golden Rule

**After ANY PR merges to main:**
1. ✅ Pull latest main
2. ✅ Create NEW feature branch
3. ✅ NEVER reuse merged branches
4. ✅ ALWAYS verify branch before commit

**Before ANY commit:**
1. ✅ Run: `git branch --show-current`
2. ✅ If shows "main" → STOP and create feature branch
3. ✅ If shows feature branch → OK to commit
4. ✅ Push to feature branch, NOT main

---

*This document was created after a Rule #5 violation on Nov 20, 2025 to ensure it never happens again.*