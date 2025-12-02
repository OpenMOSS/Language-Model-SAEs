window.initCgClerpList = function({visState, renderAll, data, cgSel}){
  let itemSel, weightValSel;
  let clerpListSel = cgSel.select('.clerp-list');

  const tokenValues = data.metadata.prompt_tokens;
  const featureById = d3.group(data.features, d => d.featureId);

  function render() {
    let nodesByTokenByLayer =  d3.group(data.nodes, d => d.ctx_idx, d => d.streamIdx);

    const finalData = Array.from(nodesByTokenByLayer.entries())
      .sort((a, b) => a[0] - b[0])
      .map(d => {
        const layers = Array.from(d[1])
        return {
          token: tokenValues[d[0]],
          values: layers.sort((a, b) => a[0] - b[0]),
        }
      })

    clerpListSel.html('')
      .st({ padding: '2px' });

    const featuresSel = clerpListSel.append('div.features')
      .st({columns: '220px', columnFill: 'auto', height: '100%'});

    const tokenSel = featuresSel.appendMany('div.token', finalData)
      .st({
        position: 'relative',
        borderTop: 'solid 1px hsl(0 0 0 / 0.4)',
      })
      .at({ title: d => d.token });

    const tokenLabelSel = tokenSel.append('div')
      .st({
        fontSize: 11,
        color: 'hsl(0 0 0 /0.4)',
        fontWeight: '400',
        pointerEvents: 'none',
        padding: '2px',
        zIndex: 1e6,
        textWrap: 'nowrap',
        overflow: 'hidden',
        textOverflow: 'ellipsis',
        textAlign: 'center',
        marginTop: 5,
      });

    tokenLabelSel.append('span').text('“');
    tokenLabelSel.append('span')
      .text(d => util.ppToken(d.token))
      .st({
        background: 'hsl(55deg 0% 85% / 0.6)',
        borderRadius: 4,
        padding: '0 2px',
        color: 'black',
        fontWeight: '700',
      });
    tokenLabelSel.append('span').text('”');

    const layerSel = tokenSel
      .appendMany('div.layer', d => d.values)
      .st({ position: 'relative' });

    const nodeSel = layerSel.appendMany('div.node', d => d[1].entries().map(d => d[1]));

    itemSel = nodeSel.append('div.feature-row')
      .classed('clicked', e => e.nodeId == visState.clickedId)
      .classed('pinned', d => visState.pinnedIds.includes(d.nodeId));

    utilCg.renderFeatureRow(itemSel, visState, renderAll);

  }
  renderAll.hClerpUpdate.fns.push(render);
  renderAll.clickedId.fns.push(render);
  renderAll.hoveredId.fns.push(() => itemSel?.classed('hovered', e => e.featureId == visState.hoveredId));
  renderAll.pinnedIds.fns.push(() => itemSel?.classed('pinned', d => visState.pinnedIds.includes(d.nodeId)));

  renderAll.clickedId.fns.push(() => {
    if (!itemSel || visState.isDev) return

    var hNode = itemSel.filter(d => d.featureId == visState.clickedId).node()
    if (!hNode) return
    var cNode = clerpListSel.node()

    var scrollTop = hNode.offsetTop - cNode.offsetTop - cNode.clientHeight / 2 + hNode.clientHeight / 2
    scrollTop = d3.clamp(0, scrollTop, cNode.scrollHeight - cNode.clientHeight)
    if (scrollTop < cNode.scrollTop - cNode.clientHeight / 2 || scrollTop > cNode.scrollTop + cNode.clientHeight / 2) {
      cNode.scrollTop = scrollTop;
    }
  });

}

window.init?.()
