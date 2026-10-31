# Complete Group Submission Archive

The complete **Group 43** Canvas ZIP is stored in three Git LFS parts because the
archive exceeds GitHub's single-file LFS limit. Together they contain:

| Requirement | Archive location |
|---|---|
| Runnable pattern recognition system and four model checkpoints | `System/` |
| 32 self-recorded videos, 6,987 sampled frames and annotations | `System/dataset/` |
| 45-page final report and editable Overleaf source | `Report/` |
| Complete 13:52.8 narrated presentation and system demonstration | `Presentations/Final_Video_Presentation.mp4` |
| First and final presentation slides | `Presentations/` |

Individual reports and peer reviews are submitted separately.

## Restore the single ZIP

Clone the repository with Git LFS installed, then run:

```sh
git lfs pull
python3 submission_assets/restore_submission.py
```

The script checks the SHA-256 of each part and the reconstructed ZIP. It writes
`deliverables/Group_43_Final_Submission_20261031.zip`. Submit that **one ZIP** to Canvas.
The resulting ZIP is about 4.64 GB. It is also available in the local project folder.

GitHub's regular source-code ZIP does not hydrate LFS objects. Use `git clone` followed
by `git lfs pull`, or download all three LFS parts individually before running the
restore script.
