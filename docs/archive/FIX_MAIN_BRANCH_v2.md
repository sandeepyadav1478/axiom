# Fix Main Branch Commit - Comprehensive Recovery

## If `git reset --hard origin/main` Didn't Work

### Option 1: Commit Was Already Pushed to Remote
If you've already pushed to remote main:

```bash
# 1. Check if commit is on remote
git log origin/main --oneline -5

# 2. If your commit is there, you need to revert on remote too
git checkout main
git reset --hard HEAD~1  # Go back one commit
git push origin main --force  # ⚠️ DANGER: Only if you're sure!

# 3. Create feature branch with your work
git checkout -b fix/test-failures-research-oct27
git cherry-pick <your-commit-hash>  # Grab your work
git push origin fix/test-failures-research-oct27
```

### Option 2: Commit Only Local, Reset Failed
If reset didn't work due to uncommitted changes:

```bash
# 1. Stash any uncommitted changes first
git stash

# 2. Now reset main
git reset --hard origin/main

# 3. Create feature branch
git checkout -b fix/test-failures-research-oct27

# 4. Apply stashed changes
git stash pop

# 5. Add and commit
git add -A
git commit -F COMMIT_MESSAGE.txt

# 6. Push feature branch
git push origin fix/test-failures-research-oct27
```

### Option 3: Start Fresh (Safest)
If above don't work:

```bash
# 1. Save your work to patch file
git diff > my_changes.patch

# 2. Reset main completely
git checkout main
git fetch origin
git reset --hard origin/main
git clean -fd  # Remove untracked files

# 3. Create fresh feature branch
git checkout -b fix/test-failures-research-terraform-oct27

# 4. Apply your changes
git apply my_changes.patch

# 5. Commit and push
git add -A
git commit -F COMMIT_MESSAGE.txt
git push origin fix/test-failures-research-terraform-oct27
```

---

## Verify Everything is Safe

```bash
# Check main is clean
git checkout main
git status
git log --oneline -3

# Check feature branch has your work
git checkout fix/test-failures-research-terraform-oct27
git log --oneline -3
```

---

## ⚠️ IMPORTANT

**NEVER use `--force` on shared branches unless you're absolutely sure!**

If you're unsure, use Option 3 (patch file) - it's safest and preserves all work.