import fs from 'node:fs';
import path from 'node:path';

const GRS_DIR = '/tmp/gh-readme-stats';

// Import the github-readme-stats handlers directly (ESM)
const { default: userStats } = await import(path.join(GRS_DIR, 'api/index.js'));
const { default: topLangs } = await import(path.join(GRS_DIR, 'api/top-langs.js'));

const noop = () => {};

/**
 * Generate an SVG by invoking the github-readme-stats handler.
 */
const generate = async (type, outPath, query) => {
  const svg = await new Promise((resolve, reject) => {
    const res = {
      setHeader: noop,
      send: (data) => resolve(data),
    };
    const handler = type === 'user' ? userStats : topLangs;
    handler({ query }, res).catch(reject);
  });

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, svg);
  console.log(`✅ Generated ${outPath} (${svg.length} bytes)`);
};

// Generate the GitHub Stats dashboard card
await generate('user', 'assets/dashboard.svg', {
  username: 'ankitranjan-dsai',
  show_icons: 'true',
  theme: 'dark',
  bg_color: '0d1117',
  title_color: '58a6ff',
  text_color: 'c9d1d9',
  icon_color: '58a6ff',
  border_color: '30363d',
  include_all_commits: 'true',
  line_height: '28',
  hide_border: 'false',
  card_width: '495',
  custom_title: 'GitHub Stats',
});

// Generate the Top Languages card
await generate('langs', 'assets/languages.svg', {
  username: 'ankitranjan-dsai',
  layout: 'compact',
  theme: 'dark',
  bg_color: '0d1117',
  title_color: '58a6ff',
  text_color: 'c9d1d9',
  icon_color: '58a6ff',
  border_color: '30363d',
  hide_border: 'false',
  langs_count: '8',
  custom_title: 'Most Used Languages',
});
