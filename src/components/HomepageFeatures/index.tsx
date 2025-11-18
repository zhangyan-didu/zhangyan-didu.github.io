import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: ReactNode;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Tech Blog',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        技术博客，记录编程学习和开发的点点滴滴。分享技术心得、解决方案和行业见解。
      </>
    ),
  },
  {
    title: 'Feelings',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        心情感悟，生活中的美好瞬间和心灵独白。记录心情随笔、读书笔记和人生感悟。
      </>
    ),
  },
  {
    title: 'Project Display',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        项目展示，展示个人作品和项目经验。从构思到实现，完整记录开发过程和技术亮点。
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  const getLinkUrl = (title: string) => {
    switch(title) {
      case 'Tech Blog':
        return '/blog';
      case 'Feelings':
        return '/feelings';
      case 'Project Display':
        return '/projects';
      default:
        return '/docs/intro';
    }
  };

  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to={getLinkUrl(title)}
          >
            进入
          </Link>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
