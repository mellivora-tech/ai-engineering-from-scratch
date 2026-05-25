#!/usr/bin/env node
const assert = require('assert');
const childProcess = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');

const root = path.resolve(__dirname, '..');
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'aifs-zh-site-'));
const outPath = path.join(tmpDir, 'data.zh.js');

childProcess.execFileSync(
  process.execPath,
  [path.join(root, 'site', 'build.js'), '--locale', 'zh', '--out', outPath],
  { cwd: root, stdio: 'pipe' }
);

assert.ok(fs.existsSync(outPath), 'expected site/build.js --out to create the requested data file');

const payload = fs.readFileSync(outPath, 'utf8');
assert.match(payload, /const SITE_LOCALE = "zh";/, 'expected generated data to record SITE_LOCALE = "zh"');
assert.match(payload, /const PHASES = \[/, 'expected generated data to include PHASES');
assert.match(payload, /const GLOSSARY = \[/, 'expected generated data to include GLOSSARY');
assert.match(payload, /"docLocale": "zh"/, 'expected translated lessons to be marked with docLocale = "zh"');
assert.match(payload, /"localDocPath": "content\/phases\/00-setup-and-tooling\/01-dev-environment\/docs\/zh.md"/, 'expected translated lessons to point at a local static markdown copy');
assert.match(payload, /"localEnglishDocPath": "content\/phases\/00-setup-and-tooling\/01-dev-environment\/docs\/en.md"/, 'expected translated lessons to point at a local English markdown copy');

const localDocPath = path.join(
  tmpDir,
  'content',
  'phases',
  '00-setup-and-tooling',
  '01-dev-environment',
  'docs',
  'zh.md'
);
assert.ok(fs.existsSync(localDocPath), 'expected zh build to copy translated markdown into content/');

const localEnglishDocPath = path.join(
  tmpDir,
  'content',
  'phases',
  '00-setup-and-tooling',
  '01-dev-environment',
  'docs',
  'en.md'
);
assert.ok(fs.existsSync(localEnglishDocPath), 'expected zh build to copy source English markdown into content/');

console.log('static zh site build test passed');
