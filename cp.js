const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch({ proxy: { server: process.env.HTTPS_PROXY, bypass: 'localhost,127.0.0.1' }, args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 1400, height: 800 }, ignoreHTTPSErrors: true });
  const errs = []; p.on('pageerror', e => errs.push('PAGE ' + String(e)));
  await p.goto('http://localhost:8766/fix.html'); await p.waitForTimeout(40000);
  console.log(await p.evaluate(() => ({ goes: document.querySelector('#goes').checked, buoys: document.querySelector('#buoys').checked, bolts: document.querySelector('#bolts').checked })));
  const bb = await (await p.$('#vsbox')).boundingBox(); await p.screenshot({ path: 'cp1.png', clip: bb, timeout: 90000 });
  await p.evaluate(() => window.__world.setMove && null); await p.evaluate(() => { const v = window.__world.viewState(); window.__world.goCam; });
  await p.keyboard.press('KeyE'); await p.keyboard.down('KeyE'); await p.waitForTimeout(700); await p.keyboard.up('KeyE'); await p.waitForTimeout(800);
  await p.screenshot({ path: 'cp2.png', clip: bb, timeout: 90000 });
  console.log(errs.join('\n') || 'no errors'); await b.close();
})();
