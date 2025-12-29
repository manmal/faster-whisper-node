const fs = require('fs');
const path = require('path');

const engineDir = path.join(__dirname, '..', 'crates', 'engine');
const libDir = path.join(__dirname, '..', 'lib_build', 'lib');

// Platform-specific library names
const platform = process.platform;
let libPatterns;

switch (platform) {
  case 'darwin':
    libPatterns = ['libctranslate2.dylib'];
    break;
  case 'linux':
    libPatterns = ['libctranslate2.so*', 'libgomp.so*'];
    break;
  case 'win32':
    libPatterns = ['*.dll'];
    break;
  default:
    console.log(`⚠️  Unknown platform: ${platform}`);
    process.exit(0);
}

// Helper to copy files (following symlinks to get the real file)
function copyLibrary(src, dstName) {
  const dst = path.join(engineDir, dstName);
  const realSrc = fs.realpathSync(src);
  fs.copyFileSync(realSrc, dst);
  console.log(`✅ Copied ${path.basename(realSrc)} -> ${dstName}`);
  return true;
}

// Copy libraries
let copied = false;
for (const pattern of libPatterns) {
  if (pattern.includes('*')) {
    // Glob pattern - list files
    const files = fs.readdirSync(libDir).filter(f => {
      const regex = new RegExp('^' + pattern.replace(/\*/g, '.*') + '$');
      return regex.test(f);
    });
    // Filter to just real files (not symlinks)
    const realFiles = files.filter(f => {
      const stat = fs.lstatSync(path.join(libDir, f));
      return !stat.isSymbolicLink();
    });
    for (const file of realFiles) {
      const src = path.join(libDir, file);
      copied = copyLibrary(src, file) || copied;
    }
  } else {
    // Exact name - follow symlinks
    const src = path.join(libDir, pattern);
    if (fs.existsSync(src)) {
      copied = copyLibrary(src, pattern) || copied;
    }
  }
}

if (!copied) {
  console.log(`⚠️  No libraries found in ${libDir}`);
}

console.log('✅ Post-build complete');
